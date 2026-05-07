#include "user.h"
#include "fcntl.h"
#include "stat.h"

#define PROMPT_MAX 4095
#define RESULT_MAX 4095
#define CONFIG_MAX_REQUESTS 64
#define CONFIG_MAX_TOKENS 255
#define TOKEN_REQ_PREFIX "@@AITOK@@"

struct test_request {
    int output_len;
    int ntokens;
    uint32 tokens[CONFIG_MAX_TOKENS];
};

static char prompt[PROMPT_MAX + 1];
static char result[RESULT_MAX + 1];
static struct test_request config_requests[CONFIG_MAX_REQUESTS];

static void append_char(char *buf, int *len, int cap, char c) {
    if (*len >= cap) {
        return;
    }
    buf[*len] = c;
    (*len)++;
    buf[*len] = '\0';
}

static void append_str(char *buf, int *len, int cap, const char *s) {
    for (int i = 0; s[i] != '\0'; i++) {
        append_char(buf, len, cap, s[i]);
    }
}

static int append_uint_text(char *buf, int *len, int cap, uint32 value) {
    char tmp[16];
    int n = 0;

    do {
        tmp[n++] = (char)('0' + (value % 10U));
        value /= 10U;
    } while (value != 0 && n < (int)sizeof(tmp));

    if (value != 0 || *len + n > cap) {
        return -1;
    }
    for (int i = n - 1; i >= 0; i--) {
        append_char(buf, len, cap, tmp[i]);
    }
    return 0;
}

static int read_full_local(int fd, char *buf, int n) {
    int done = 0;
    while (done < n) {
        int r = read(fd, buf + done, n - done);
        if (r <= 0) {
            return -1;
        }
        done += r;
    }
    return 0;
}

static int load_text_file(const char *path, char **buf_out) {
    struct stat st;
    if (stat(path, &st) < 0 || st.size > 0x7fffffffULL) {
        return -1;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    char *buf = (char *)sbrk((int)st.size + 1);
    if (buf == (char *)-1) {
        close(fd);
        return -1;
    }
    if (st.size > 0 && read_full_local(fd, buf, (int)st.size) < 0) {
        close(fd);
        return -1;
    }
    close(fd);
    buf[st.size] = '\0';
    *buf_out = buf;
    return (int)st.size;
}

static void skip_ws(const char **ps) {
    while (**ps == ' ' || **ps == '\t' || **ps == '\n' || **ps == '\r') {
        (*ps)++;
    }
}

static int consume_char(const char **ps, char ch) {
    skip_ws(ps);
    if (**ps != ch) {
        return -1;
    }
    (*ps)++;
    return 0;
}

static int parse_json_string(const char **ps, char *out, int cap) {
    skip_ws(ps);
    if (**ps != '"' || cap <= 0) {
        return -1;
    }
    (*ps)++;

    int len = 0;
    while (**ps != '\0' && **ps != '"') {
        if (**ps == '\\') {
            return -1;
        }
        if (len + 1 >= cap) {
            return -1;
        }
        out[len++] = **ps;
        (*ps)++;
    }
    if (**ps != '"') {
        return -1;
    }
    (*ps)++;
    out[len] = '\0';
    return 0;
}

static int parse_json_uint32(const char **ps, uint32 *out) {
    uint64 v = 0;
    int saw_digit = 0;

    skip_ws(ps);
    while (**ps >= '0' && **ps <= '9') {
        v = v * 10ULL + (uint64)(**ps - '0');
        if (v > 0xffffffffULL) {
            return -1;
        }
        (*ps)++;
        saw_digit = 1;
    }
    if (!saw_digit) {
        return -1;
    }
    *out = (uint32)v;
    return 0;
}

static int parse_json_positive_int(const char **ps, int *out) {
    uint32 v = 0;
    if (parse_json_uint32(ps, &v) < 0 || v == 0 || v > 0x7fffffffU) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int is_output_key(const char *key) {
    return strcmp(key, "output_len") == 0 || strcmp(key, "output_length") == 0;
}

static int is_token_key(const char *key) {
    return strcmp(key, "input_token_ids") == 0 || strcmp(key, "token_ids") == 0 || strcmp(key, "input_ids") == 0;
}

static int parse_token_array(const char **ps, struct test_request *req) {
    if (consume_char(ps, '[') < 0) {
        return -1;
    }
    skip_ws(ps);
    if (**ps == ']') {
        return -1;
    }

    int count = 0;
    for (;;) {
        if (count >= CONFIG_MAX_TOKENS) {
            return -1;
        }
        if (parse_json_uint32(ps, &req->tokens[count]) < 0) {
            return -1;
        }
        count++;

        skip_ws(ps);
        if (**ps == ',') {
            (*ps)++;
            continue;
        }
        if (**ps == ']') {
            (*ps)++;
            break;
        }
        return -1;
    }

    req->ntokens = count;
    return 0;
}

static int parse_request_object(const char **ps, struct test_request *req) {
    char key[32];
    int saw_output = 0;
    int saw_tokens = 0;

    memset(req, 0, sizeof(*req));
    if (consume_char(ps, '{') < 0) {
        return -1;
    }

    skip_ws(ps);
    if (**ps == '}') {
        return -1;
    }

    for (;;) {
        if (parse_json_string(ps, key, sizeof(key)) < 0) {
            return -1;
        }
        if (consume_char(ps, ':') < 0) {
            return -1;
        }

        if (is_output_key(key)) {
            if (saw_output || parse_json_positive_int(ps, &req->output_len) < 0 || req->output_len > CONFIG_MAX_TOKENS) {
                return -1;
            }
            saw_output = 1;
        } else if (is_token_key(key)) {
            if (saw_tokens || parse_token_array(ps, req) < 0) {
                return -1;
            }
            saw_tokens = 1;
        } else {
            return -1;
        }

        skip_ws(ps);
        if (**ps == ',') {
            (*ps)++;
            continue;
        }
        if (**ps == '}') {
            (*ps)++;
            break;
        }
        return -1;
    }

    if (!saw_output || !saw_tokens || req->ntokens <= 0) {
        return -1;
    }
    return 0;
}

static int parse_request_array(const char **ps, int *count_out) {
    if (consume_char(ps, '[') < 0) {
        return -1;
    }
    skip_ws(ps);
    if (**ps == ']') {
        return -1;
    }

    int count = 0;
    for (;;) {
        if (count >= CONFIG_MAX_REQUESTS) {
            return -1;
        }
        if (parse_request_object(ps, &config_requests[count]) < 0) {
            return -1;
        }
        count++;

        skip_ws(ps);
        if (**ps == ',') {
            (*ps)++;
            continue;
        }
        if (**ps == ']') {
            (*ps)++;
            break;
        }
        return -1;
    }

    *count_out = count;
    return 0;
}

static int parse_config_text(const char *text, int *count_out) {
    const char *p = text;
    char key[32];

    skip_ws(&p);
    if (*p == '[') {
        if (parse_request_array(&p, count_out) < 0) {
            return -1;
        }
    } else if (*p == '{') {
        if (consume_char(&p, '{') < 0) {
            return -1;
        }
        if (parse_json_string(&p, key, sizeof(key)) < 0 || strcmp(key, "requests") != 0) {
            return -1;
        }
        if (consume_char(&p, ':') < 0 || parse_request_array(&p, count_out) < 0) {
            return -1;
        }
        if (consume_char(&p, '}') < 0) {
            return -1;
        }
    } else {
        return -1;
    }

    skip_ws(&p);
    return *p == '\0' ? 0 : -1;
}

static int encode_token_request(const struct test_request *req, char *buf, int cap) {
    int len = 0;

    buf[0] = '\0';
    append_str(buf, &len, cap, TOKEN_REQ_PREFIX);
    if (append_uint_text(buf, &len, cap, (uint32)req->output_len) < 0) {
        return -1;
    }
    append_char(buf, &len, cap, ';');
    for (int i = 0; i < req->ntokens; i++) {
        if (i > 0) {
            append_char(buf, &len, cap, ',');
        }
        if (append_uint_text(buf, &len, cap, req->tokens[i]) < 0) {
            return -1;
        }
    }
    return len;
}

static void run_config_requests(const char *config_path, int use_async, int do_query) {
    char *config_text = 0;
    int nreq = 0;

    if (load_text_file(config_path, &config_text) < 0) {
        fprintf(2, "aitest: failed to read config %s\n", config_path);
        exit(1);
    }
    if (parse_config_text(config_text, &nreq) < 0 || nreq <= 0) {
        fprintf(2, "aitest: bad config JSON\n");
        exit(1);
    }

    for (int i = 0; i < nreq; i++) {
        memset(prompt, 0, sizeof(prompt));
        memset(result, 0, sizeof(result));
        int prompt_len = encode_token_request(&config_requests[i], prompt, PROMPT_MAX);
        if (prompt_len < 0) {
            fprintf(2, "aitest: encoded request too large at index %d\n", i);
            exit(1);
        }

        int n = -1;
        if (use_async) {
            int reqid = ai_submit(prompt, prompt_len);
            if (reqid < 0) {
                fprintf(2, "aitest: ai_submit failed at request %d\n", i);
                exit(1);
            }
            printf("[aitest] request[%d] reqid=%d output_len=%d input_tokens=%d\n",
                   i, reqid, config_requests[i].output_len, config_requests[i].ntokens);

            if (do_query) {
                struct ai_status st;
                if (ai_query(reqid, &st) < 0) {
                    fprintf(2, "aitest: ai_query failed at request %d\n", i);
                    exit(1);
                }
                printf("[aitest] request[%d] query: reqid=%d state=%d err=%d result_len=%d\n",
                       i, st.reqid, st.state, st.err, st.result_len);
            }

            n = ai_wait(reqid, result, sizeof(result));
            if (n < 0) {
                fprintf(2, "aitest: ai_wait failed at request %d\n", i);
                exit(1);
            }
        } else {
            n = ai_call(prompt, prompt_len, result, sizeof(result));
            if (n < 0) {
                fprintf(2, "aitest: ai_call failed at request %d\n", i);
                exit(1);
            }
            printf("[aitest] request[%d] output_len=%d input_tokens=%d\n",
                   i, config_requests[i].output_len, config_requests[i].ntokens);
        }

        printf("[aitest] request[%d] result: %s\n", i, result);
        printf("[aitest] request[%d] bytes: %d\n", i, n);
    }

    exit(0);
}

int main(int argc, char **argv) {
    int use_async = 0;
    int badwait = 0;
    int batch = 0;
    int do_query = 0;
    int doublewait = 0;
    int fork_unauth = 0;
    char *config_path = 0;
    memset(prompt, 0, sizeof(prompt));

    int argi = 1;
    while (argi < argc) {
        if (strcmp(argv[argi], "--async") == 0) {
            use_async = 1;
            argi++;
            continue;
        }
        if (strcmp(argv[argi], "--badwait") == 0) {
            badwait = 1;
            argi++;
            continue;
        }
        if (strcmp(argv[argi], "--query") == 0) {
            do_query = 1;
            argi++;
            continue;
        }
        if (strcmp(argv[argi], "--doublewait") == 0) {
            doublewait = 1;
            argi++;
            continue;
        }
        if (strcmp(argv[argi], "--fork-unauth") == 0) {
            fork_unauth = 1;
            argi++;
            continue;
        }
        if (strcmp(argv[argi], "--batch") == 0 && argi + 1 < argc) {
            batch = atoi(argv[argi + 1]);
            argi += 2;
            continue;
        }
        if (strcmp(argv[argi], "--config") == 0 && argi + 1 < argc) {
            config_path = argv[argi + 1];
            argi += 2;
            continue;
        }
        break;
    }

    if (config_path != 0) {
        if (badwait || batch > 0 || doublewait || fork_unauth || argc > argi) {
            fprintf(2, "aitest: --config only supports sequential requests with optional --async/--query\n");
            exit(1);
        }
        run_config_requests(config_path, use_async, do_query);
    }

    if (argc <= argi) {
        strcpy(prompt, "Answer in one short phrase. The capital of France is");
    } else {
        int len = 0;
        for (int i = argi; i < argc; i++) {
            if (i > argi) {
                append_char(prompt, &len, PROMPT_MAX, ' ');
            }
            for (int j = 0; argv[i][j] != '\0'; j++) {
                append_char(prompt, &len, PROMPT_MAX, argv[i][j]);
            }
        }
    }

    memset(result, 0, sizeof(result));

    if (badwait) {
        int r = ai_wait(9999, result, sizeof(result));
        if (r >= 0) {
            fprintf(2, "aitest: ai_wait on bad reqid unexpectedly succeeded\n");
            exit(1);
        }
        printf("[aitest] badwait: PASS\n");
        exit(0);
    }

    if (batch > 0) {
        if (batch > 8) {
            fprintf(2, "aitest: batch too large\n");
            exit(1);
        }
        int reqids[8];
        for (int i = 0; i < batch; i++) {
            reqids[i] = ai_submit(prompt, (int)strlen(prompt));
            if (reqids[i] < 0) {
                fprintf(2, "aitest: ai_submit failed at batch %d\n", i);
                exit(1);
            }
            printf("[aitest] batch submit[%d] reqid=%d\n", i, reqids[i]);
        }
        for (int i = 0; i < batch; i++) {
            memset(result, 0, sizeof(result));
            int n = ai_wait(reqids[i], result, sizeof(result));
            if (n < 0) {
                fprintf(2, "aitest: ai_wait failed at batch %d\n", i);
                exit(1);
            }
            printf("[aitest] batch result[%d]: %s (%d bytes)\n", i, result, n);
        }
        exit(0);
    }

    int n;
    if (use_async) {
        int reqid = ai_submit(prompt, (int)strlen(prompt));
        if (reqid < 0) {
            fprintf(2, "aitest: ai_submit failed\n");
            exit(1);
        }
        printf("[aitest] reqid: %d\n", reqid);

        if (do_query) {
            struct ai_status st;
            if (ai_query(reqid, &st) < 0) {
                fprintf(2, "aitest: ai_query failed\n");
                exit(1);
            }
            printf("[aitest] query: reqid=%d state=%d err=%d result_len=%d\n",
                   st.reqid, st.state, st.err, st.result_len);
        }

        if (fork_unauth) {
            int pid = fork();
            if (pid < 0) {
                fprintf(2, "aitest: fork failed\n");
                exit(1);
            }
            if (pid == 0) {
                int r = ai_wait(reqid, result, sizeof(result));
                exit(r < 0 ? 0 : 1);
            }
            int status = -1;
            if (wait(&status) < 0 || status != 0) {
                fprintf(2, "aitest: child unauthorized wait check failed\n");
                exit(1);
            }
            printf("[aitest] fork-unauth: PASS\n");
        }

        n = ai_wait(reqid, result, sizeof(result));
        if (n < 0) {
            fprintf(2, "aitest: ai_wait failed\n");
            exit(1);
        }

        if (doublewait) {
            int r = ai_wait(reqid, result, sizeof(result));
            if (r >= 0) {
                fprintf(2, "aitest: second ai_wait unexpectedly succeeded\n");
                exit(1);
            }
            printf("[aitest] doublewait: PASS\n");
        }
    } else {
        n = ai_call(prompt, (int)strlen(prompt), result, sizeof(result));
        if (n < 0) {
            fprintf(2, "aitest: ai_call failed\n");
            exit(1);
        }
    }

    printf("[aitest] prompt: %s\n", prompt);
    printf("[aitest] result: %s\n", result);
    printf("[aitest] bytes: %d\n", n);
    exit(0);
}
