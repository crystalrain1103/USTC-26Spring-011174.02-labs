#include "user.h"
#include "fcntl.h"
#include "stat.h"

#define TOKEN_MAX 255
#define PREDICT_MAX 32
#define RESULT_MAX 255
#define TOKEN_TEXT_MAX 3072
#define DEFAULT_TOKEN_COUNT 1
#define DEFAULT_TOKEN_VALUE 0
#define DEFAULT_PREDICT_COUNT 1

static char token_text[TOKEN_TEXT_MAX];

static int read_full(int fd, void *buf, int n) {
    char *p = (char *)buf;
    int done = 0;
    while (done < n) {
        int r = read(fd, p + done, n - done);
        if (r <= 0) {
            return -1;
        }
        done += r;
    }
    return 0;
}

static int active_model_file(const char *name, char *out, int out_cap) {
    struct stat st;
    char path[32];
    memset(path, 0, sizeof(path));
    strcpy(path, "/AI/SMOL/");
    if ((int)strlen(path) + (int)strlen(name) + 1 <= (int)sizeof(path)) {
        strcpy(path + strlen(path), name);
    }
    if (stat(path, &st) == 0) {
        if ((int)strlen(path) + 1 > out_cap) {
            return -1;
        }
        strcpy(out, path);
        return 0;
    }

    memset(path, 0, sizeof(path));
    strcpy(path, "/AI/QWEN/");
    if ((int)strlen(path) + (int)strlen(name) + 1 <= (int)sizeof(path)) {
        strcpy(path + strlen(path), name);
    }
    if (stat(path, &st) == 0) {
        if ((int)strlen(path) + 1 > out_cap) {
            return -1;
        }
        strcpy(out, path);
        return 0;
    }
    return -1;
}

static int load_u32_file(const char *path, uint32 *tokens, int cap) {
    char raw[4 + TOKEN_MAX * sizeof(uint32)];
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0 || st.size < 4 || st.size > sizeof(raw)) {
        close(fd);
        return -1;
    }
    if (read_full(fd, raw, (int)st.size) < 0) {
        close(fd);
        return -1;
    }
    close(fd);

    uint32 count = *(uint32 *)raw;
    if ((int)count <= 0 || (int)count > cap || st.size != 4ULL + (uint64)count * sizeof(uint32)) {
        return -1;
    }
    memmove(tokens, raw + 4, (int)count * sizeof(uint32));
    return (int)count;
}

static int load_default_tokens(uint32 *tokens, int cap) {
    char path[32];
    if (active_model_file("PMT.BIN", path, sizeof(path)) < 0) {
        if (cap < DEFAULT_TOKEN_COUNT) {
            return -1;
        }
        tokens[0] = DEFAULT_TOKEN_VALUE;
        return DEFAULT_TOKEN_COUNT;
    }
    return load_u32_file(path, tokens, cap);
}

static int load_default_predict(void) {
    char path[32];
    uint32 count_and_vals[1 + PREDICT_MAX];
    if (active_model_file("EXP.BIN", path, sizeof(path)) < 0) {
        return DEFAULT_PREDICT_COUNT;
    }
    return load_u32_file(path, count_and_vals, PREDICT_MAX);
}

static int format_tokens(const uint32 *tokens, int token_count, char *out, int out_cap) {
    int pos = 0;
    for (int i = 0; i < token_count; i++) {
        char digits[16];
        int nd = 0;
        uint32 value = tokens[i];

        if (i > 0) {
            if (pos + 1 >= out_cap) {
                return -1;
            }
            out[pos++] = ' ';
        }

        do {
            digits[nd++] = (char)('0' + (value % 10));
            value /= 10;
        } while (value != 0 && nd < (int)sizeof(digits));

        if (pos + nd >= out_cap) {
            return -1;
        }
        for (int j = nd - 1; j >= 0; j--) {
            out[pos++] = digits[j];
        }
    }
    if (pos >= out_cap) {
        return -1;
    }
    out[pos] = '\0';
    return pos;
}

static int run_cacheprobe(const uint32 *tokens, int token_count, int predict_count, char *out, int out_cap) {
    char cold[RESULT_MAX + 1];
    int cold_start = 0;
    int warm_start = 0;
    int cold_ticks = 0;
    int warm_ticks = 0;
    int cold_n = 0;
    int warm_n = 0;

    memset(cold, 0, sizeof(cold));
    memset(out, 0, out_cap);

    printf("[aitest] cacheprobe: cold run\n");
    cold_start = uptime();
    cold_n = ai_call(tokens, token_count, predict_count, cold, sizeof(cold));
    cold_ticks = uptime() - cold_start;
    if (cold_n < 0) {
        fprintf(2, "aitest: cacheprobe cold ai_call failed\n");
        return -1;
    }
    printf("[aitest] cacheprobe cold ticks: %d\n", cold_ticks);

    printf("[aitest] cacheprobe: warm run\n");
    warm_start = uptime();
    warm_n = ai_call(tokens, token_count, predict_count, out, out_cap);
    warm_ticks = uptime() - warm_start;
    if (warm_n < 0) {
        fprintf(2, "aitest: cacheprobe warm ai_call failed\n");
        return -1;
    }
    printf("[aitest] cacheprobe warm ticks: %d\n", warm_ticks);

    if (strcmp(cold, out) != 0) {
        printf("[aitest] cacheprobe note: result mismatch between cold and warm runs\n");
    }
    return warm_n;
}

int main(int argc, char **argv) {
    int use_async = 0;
    int badwait = 0;
    int batch = 0;
    int cacheprobe = 0;
    int do_query = 0;
    int doublewait = 0;
    int fork_unauth = 0;
    int predict_count = -1;
    uint32 tokens[TOKEN_MAX];
    int token_count = 0;
    memset(tokens, 0, sizeof(tokens));

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
        if (strcmp(argv[argi], "--cacheprobe") == 0) {
            cacheprobe = 1;
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
        if (strcmp(argv[argi], "--predict") == 0 && argi + 1 < argc) {
            predict_count = atoi(argv[argi + 1]);
            argi += 2;
            continue;
        }
        if (strcmp(argv[argi], "--batch") == 0 && argi + 1 < argc) {
            batch = atoi(argv[argi + 1]);
            argi += 2;
            continue;
        }
        break;
    }

    if (argc <= argi) {
        token_count = load_default_tokens(tokens, TOKEN_MAX);
        if (token_count <= 0) {
            fprintf(2, "aitest: failed to load default prompt tokens\n");
            exit(1);
        }
    } else {
        for (int i = argi; i < argc; i++) {
            int value = atoi(argv[i]);
            if (value < 0 || token_count >= TOKEN_MAX) {
                fprintf(2, "aitest: invalid token id %s\n", argv[i]);
                exit(1);
            }
            tokens[token_count++] = (uint32)value;
        }
    }
    if (token_count <= 0) {
        fprintf(2, "aitest: no tokens supplied\n");
        exit(1);
    }
    if (predict_count < 0) {
        predict_count = load_default_predict();
        if (predict_count <= 0) {
            fprintf(2, "aitest: failed to load default predict count\n");
            exit(1);
        }
    }
    if (predict_count <= 0 || predict_count > PREDICT_MAX) {
        fprintf(2, "aitest: invalid predict count %d\n", predict_count);
        exit(1);
    }

    char result[RESULT_MAX + 1];
    memset(result, 0, sizeof(result));
    memset(token_text, 0, sizeof(token_text));
    if (format_tokens(tokens, token_count, token_text, sizeof(token_text)) < 0) {
        fprintf(2, "aitest: token list too long to print\n");
        exit(1);
    }
    if (cacheprobe && (use_async || badwait || batch > 0 || do_query || doublewait || fork_unauth)) {
        fprintf(2, "aitest: --cacheprobe only supports the simple synchronous path\n");
        exit(1);
    }

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
            reqids[i] = ai_submit(tokens, token_count, predict_count);
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
    if (cacheprobe) {
        n = run_cacheprobe(tokens, token_count, predict_count, result, sizeof(result));
        if (n < 0) {
            exit(1);
        }
    } else if (use_async) {
        int reqid = ai_submit(tokens, token_count, predict_count);
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
        n = ai_call(tokens, token_count, predict_count, result, sizeof(result));
        if (n < 0) {
            fprintf(2, "aitest: ai_call failed\n");
            exit(1);
        }
    }

    printf("[aitest] tokens: %s\n", token_text);
    printf("[aitest] token_count: %d\n", token_count);
    printf("[aitest] predict_count: %d\n", predict_count);
    printf("[aitest] result: %s\n", result);
    printf("[aitest] bytes: %d\n", n);
    exit(0);
}
