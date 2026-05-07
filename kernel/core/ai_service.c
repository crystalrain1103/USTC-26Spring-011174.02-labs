#include "types.h"
#include "defs.h"
#include "proc.h"
#include "spinlock.h"

#define AI_NREQ 8
#define AI_NWORKER 2
#define AI_MAX_PROMPT 4095
#define AI_MAX_RESULT 4095

static const char ai_token_req_prefix[] = "@@AITOK@@";

enum ai_req_state {
    AIREQ_UNUSED = 0,
    AIREQ_NEW,
    AIREQ_READY,
    AIREQ_RUNNING,
    AIREQ_DONE,
    AIREQ_FAILED,
};

struct ai_request {
    int id;
    int owner_pid;
    int state;
    int err;
    int prompt_len;
    int result_len;
    char prompt[AI_MAX_PROMPT + 1];
    char result[AI_MAX_RESULT + 1];
};

struct ai_service {
    struct spinlock lock;
    int next_id;
    int q[AI_NREQ];
    int qhead;
    int qtail;
    int qcount;
    struct ai_request reqs[AI_NREQ];
} aisvc;

void ai_service_worker(void);

static void ai_req_reset(struct ai_request *req) {
    memset(req, 0, sizeof(*req));
    req->state = AIREQ_UNUSED;
}

static int ai_req_busy(const struct ai_request *req) {
    return req->state == AIREQ_NEW || req->state == AIREQ_READY || req->state == AIREQ_RUNNING;
}

static struct ai_request *ai_find_slot_locked(void) {
    for (int i = 0; i < AI_NREQ; i++) {
        if (aisvc.reqs[i].state == AIREQ_UNUSED) {
            return &aisvc.reqs[i];
        }
    }
    return 0;
}

static struct ai_request *ai_find_req_locked(int reqid, int owner_pid) {
    for (int i = 0; i < AI_NREQ; i++) {
        struct ai_request *req = &aisvc.reqs[i];
        if (req->state != AIREQ_UNUSED && req->id == reqid && req->owner_pid == owner_pid) {
            return req;
        }
    }
    return 0;
}

static int kstrcmp_local(const char *a, const char *b) {
    int i = 0;
    while (a[i] != '\0' && b[i] != '\0') {
        if (a[i] != b[i]) {
            return (int)((uchar)a[i]) - (int)((uchar)b[i]);
        }
        i++;
    }
    return (int)((uchar)a[i]) - (int)((uchar)b[i]);
}

static int kstrncmp_local(const char *a, const char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return (int)((uchar)a[i]) - (int)((uchar)b[i]);
        }
        if (a[i] == '\0') {
            return 0;
        }
    }
    return 0;
}

static int ai_parse_positive_int(const char **ps, int *out) {
    const char *s = *ps;
    int v = 0;
    int saw_digit = 0;

    while (*s >= '0' && *s <= '9') {
        int digit = *s - '0';
        if (v > (0x7fffffff - digit) / 10) {
            return -1;
        }
        v = v * 10 + digit;
        s++;
        saw_digit = 1;
    }
    if (!saw_digit || v <= 0) {
        return -1;
    }

    *ps = s;
    *out = v;
    return 0;
}

static int ai_validate_token_csv(const char *s) {
    int saw_token = 0;

    if (s == 0 || *s == '\0') {
        return -1;
    }

    for (;;) {
        int saw_digit = 0;
        while (*s >= '0' && *s <= '9') {
            s++;
            saw_digit = 1;
        }
        if (!saw_digit) {
            return -1;
        }
        saw_token = 1;
        if (*s == '\0') {
            return saw_token ? 0 : -1;
        }
        if (*s != ',') {
            return -1;
        }
        s++;
        if (*s == '\0') {
            return -1;
        }
    }
}

static int ai_format_positive_int(int value, char *buf, int cap) {
    char tmp[16];
    int n = 0;

    if (buf == 0 || cap <= 1 || value <= 0) {
        return -1;
    }
    while (value > 0 && n < (int)sizeof(tmp)) {
        tmp[n++] = (char)('0' + (value % 10));
        value /= 10;
    }
    if (value != 0 || n + 1 > cap) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        buf[i] = tmp[n - 1 - i];
    }
    buf[n] = '\0';
    return n;
}

static void ai_make_result_path(int pid, char *out, int cap) {
    char digits[16];
    int nd = 0;
    int x = pid;
    if (out == 0 || cap < 12 || pid <= 0) {
        panic("ai_make_result_path");
    }
    while (x > 0 && nd < (int)sizeof(digits)) {
        digits[nd++] = (char)('0' + (x % 10));
        x /= 10;
    }
    int p = 0;
    out[p++] = '/';
    out[p++] = 'A';
    out[p++] = 'I';
    out[p++] = '.';
    for (int i = nd - 1; i >= 0; i--) {
        if (p >= cap - 1) {
            panic("ai_make_result_path overflow");
        }
        out[p++] = digits[i];
    }
    out[p++] = '.';
    out[p++] = 'O';
    out[p++] = 'U';
    out[p++] = 'T';
    out[p] = '\0';
}

static int ai_read_file_text(const char *path, char *buf, int cap) {
    if (path == 0 || buf == 0 || cap <= 1) {
        return -1;
    }
    struct inode *ip = namei(path);
    if (ip == 0) {
        return -1;
    }
    int n = readi(ip, 0, buf, (uint)(cap - 1));
    iput(ip);
    if (n < 0) {
        return -1;
    }
    buf[n] = '\0';
    return n;
}

static int ai_extract_prompt_text(char *dst, int cap) {
    char raw[512];
    int n = ai_read_file_text("/PROMPT.TXT", raw, sizeof(raw));
    if (n < 0) {
        return -1;
    }

    const char prefix[] = "prompt_text = \"";
    int i = 0;
    while (prefix[i] != '\0') {
        if (raw[i] != prefix[i]) {
            return -1;
        }
        i++;
    }
    int j = 0;
    while (raw[i] != '\0' && raw[i] != '"' && j + 1 < cap) {
        dst[j++] = raw[i++];
    }
    dst[j] = '\0';
    return j;
}

static int ai_run_real_llm(const char *prompt, char *out, int out_cap) {
    if (prompt == 0 || out == 0 || out_cap <= 1) {
        return -1;
    }

    int prefix_len = (int)(sizeof(ai_token_req_prefix) - 1);
    if (kstrncmp_local(prompt, ai_token_req_prefix, prefix_len) == 0) {
        const char *p = prompt + prefix_len;
        int gen_len = 0;
        if (ai_parse_positive_int(&p, &gen_len) < 0 || *p != ';') {
            return -1;
        }
        const char *token_csv = p + 1;
        if (ai_validate_token_csv(token_csv) < 0) {
            return -1;
        }

        char gen_arg[16];
        if (ai_format_positive_int(gen_len, gen_arg, sizeof(gen_arg)) < 0) {
            return -1;
        }

        char *argv[] = {
            "llmrun",
            "--prompt-tokens",
            (char *)token_csv,
            "--gen-len",
            gen_arg,
            0,
        };

        int pid = proc_spawn_user_service_argv("/llmrun", argv);
        if (pid < 0) {
            return -1;
        }

        int status = -1;
        if (proc_wait_pid(pid, &status) < 0 || status != 0) {
            return -1;
        }

        char result_path[32];
        ai_make_result_path(pid, result_path, sizeof(result_path));
        int n = ai_read_file_text(result_path, out, out_cap);
        if (n < 0) {
            return -1;
        }
        while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r')) {
            out[n - 1] = '\0';
            n--;
        }
        return n;
    }

    char expected_prompt[AI_MAX_PROMPT + 1];
    if (ai_extract_prompt_text(expected_prompt, sizeof(expected_prompt)) < 0) {
        return -1;
    }
    if (kstrcmp_local(prompt, expected_prompt) != 0) {
        return -1;
    }

    int pid = proc_spawn_user_service("/llmrun");
    if (pid < 0) {
        return -1;
    }

    int status = -1;
    if (proc_wait_pid(pid, &status) < 0 || status != 0) {
        return -1;
    }

    char result_path[32];
    ai_make_result_path(pid, result_path, sizeof(result_path));
    int n = ai_read_file_text(result_path, out, out_cap);
    if (n < 0) {
        return -1;
    }
    while (n > 0 && (out[n - 1] == '\n' || out[n - 1] == '\r')) {
        out[n - 1] = '\0';
        n--;
    }
    return n;
}

void ai_service_init(void) {
    initlock(&aisvc.lock, "ai_service");
    aisvc.next_id = 1;
    aisvc.qhead = 0;
    aisvc.qtail = 0;
    aisvc.qcount = 0;
    for (int i = 0; i < AI_NREQ; i++) {
        ai_req_reset(&aisvc.reqs[i]);
        aisvc.q[i] = -1;
    }
    for (int i = 0; i < AI_NWORKER; i++) {
        if (proc_create(ai_service_worker) < 0) {
            panic("ai_service_init: worker");
        }
    }
}

static int ai_service_enqueue_prompt(uint64 prompt_uva, int prompt_len, int *reqid_out) {
    struct proc *p = myproc();
    if (p == 0 || p->pagetable == 0) {
        return -1;
    }
    if (prompt_len <= 0 || prompt_len > AI_MAX_PROMPT) {
        return -1;
    }

    char prompt[AI_MAX_PROMPT + 1];
    memset(prompt, 0, sizeof(prompt));
    if (copyin(p->pagetable, prompt, prompt_uva, (uint64)prompt_len) < 0) {
        return -1;
    }
    prompt[prompt_len] = '\0';

    acquire(&aisvc.lock);
    struct ai_request *req = 0;
    while ((req = ai_find_slot_locked()) == 0) {
        sleep(&aisvc, &aisvc.lock);
    }

    ai_req_reset(req);
    req->id = aisvc.next_id++;
    req->owner_pid = p->pid;
    req->err = 0;
    req->prompt_len = prompt_len;
    req->state = AIREQ_NEW;
    memmove(req->prompt, prompt, (uint)prompt_len + 1);
    req->state = AIREQ_READY;
    int idx = (int)(req - aisvc.reqs);
    aisvc.q[aisvc.qtail] = idx;
    aisvc.qtail = (aisvc.qtail + 1) % AI_NREQ;
    aisvc.qcount++;
    wakeup(&aisvc.qcount);
    if (reqid_out) {
        *reqid_out = req->id;
    }
    release(&aisvc.lock);
    return 0;
}

void ai_service_worker(void) {
    for (;;) {
        acquire(&aisvc.lock);
        while (aisvc.qcount == 0) {
            sleep(&aisvc.qcount, &aisvc.lock);
        }
        int idx = aisvc.q[aisvc.qhead];
        aisvc.q[aisvc.qhead] = -1;
        aisvc.qhead = (aisvc.qhead + 1) % AI_NREQ;
        aisvc.qcount--;
        struct ai_request *req = &aisvc.reqs[idx];
        req->state = AIREQ_RUNNING;
        char prompt[AI_MAX_PROMPT + 1];
        memmove(prompt, req->prompt, (uint)req->prompt_len + 1);
        release(&aisvc.lock);

        char result[AI_MAX_RESULT + 1];
        memset(result, 0, sizeof(result));
        int result_len = ai_run_real_llm(prompt, result, sizeof(result));

        acquire(&aisvc.lock);
        if (result_len < 0) {
            req->err = -1;
            req->result_len = 0;
            req->state = AIREQ_FAILED;
        } else {
            req->err = 0;
            req->result_len = result_len;
            memmove(req->result, result, (uint)result_len + 1);
            req->state = AIREQ_DONE;
        }
        wakeup(req);
        release(&aisvc.lock);
    }
}

int ai_service_submit(uint64 prompt_uva, int prompt_len) {
    int reqid = -1;
    if (ai_service_enqueue_prompt(prompt_uva, prompt_len, &reqid) < 0) {
        return -1;
    }
    return reqid;
}

int ai_service_query(int reqid, uint64 st_uva) {
    struct proc *p = myproc();
    if (p == 0 || p->pagetable == 0 || reqid <= 0) {
        return -1;
    }

    acquire(&aisvc.lock);
    struct ai_request *req = ai_find_req_locked(reqid, p->pid);
    if (req == 0) {
        release(&aisvc.lock);
        return -1;
    }

    struct ai_status st;
    st.reqid = req->id;
    st.state = req->state;
    st.err = req->err;
    st.result_len = req->result_len;
    release(&aisvc.lock);

    if (copyout(p->pagetable, st_uva, (char *)&st, sizeof(st)) < 0) {
        return -1;
    }
    return 0;
}

int ai_service_wait(int reqid, uint64 out_uva, int out_cap) {
    struct proc *p = myproc();
    if (p == 0 || p->pagetable == 0 || reqid <= 0 || out_cap <= 0 || out_cap > AI_MAX_RESULT + 1) {
        return -1;
    }

    acquire(&aisvc.lock);
    struct ai_request *req = ai_find_req_locked(reqid, p->pid);
    if (req == 0) {
        release(&aisvc.lock);
        return -1;
    }
    while (ai_req_busy(req)) {
        sleep(req, &aisvc.lock);
    }

    if (req->state != AIREQ_DONE || req->err < 0) {
        ai_req_reset(req);
        wakeup(&aisvc);
        release(&aisvc.lock);
        return -1;
    }

    int n = req->result_len;
    if (out_cap < n + 1) {
        ai_req_reset(req);
        wakeup(&aisvc);
        release(&aisvc.lock);
        return -1;
    }

    char result[AI_MAX_RESULT + 1];
    memmove(result, req->result, (uint)n + 1);
    ai_req_reset(req);
    wakeup(&aisvc);
    release(&aisvc.lock);

    if (copyout(p->pagetable, out_uva, result, (uint64)n + 1) < 0) {
        return -1;
    }
    return n;
}

int ai_service_call(uint64 prompt_uva, int prompt_len, uint64 out_uva, int out_cap) {
    int reqid = ai_service_submit(prompt_uva, prompt_len);
    if (reqid < 0) {
        return -1;
    }
    return ai_service_wait(reqid, out_uva, out_cap);
}
