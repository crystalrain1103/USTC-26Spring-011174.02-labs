#include "user.h"
#include "syscall.h"

static inline long __syscall(long num, long a0, long a1, long a2) {
    register long _a0 asm("a0") = a0;
    register long _a1 asm("a1") = a1;
    register long _a2 asm("a2") = a2;
    register long _a7 asm("a7") = num;
    asm volatile("ecall" : "+r"(_a0) : "r"(_a1), "r"(_a2), "r"(_a7) : "memory");
    return _a0;
}

static inline long __syscall4(long num, long a0, long a1, long a2, long a3) {
    register long _a0 asm("a0") = a0;
    register long _a1 asm("a1") = a1;
    register long _a2 asm("a2") = a2;
    register long _a3 asm("a3") = a3;
    register long _a7 asm("a7") = num;
    asm volatile("ecall" : "+r"(_a0) : "r"(_a1), "r"(_a2), "r"(_a3), "r"(_a7) : "memory");
    return _a0;
}

static inline long __syscall5(long num, long a0, long a1, long a2, long a3, long a4) {
    register long _a0 asm("a0") = a0;
    register long _a1 asm("a1") = a1;
    register long _a2 asm("a2") = a2;
    register long _a3 asm("a3") = a3;
    register long _a4 asm("a4") = a4;
    register long _a7 asm("a7") = num;
    asm volatile("ecall" : "+r"(_a0) : "r"(_a1), "r"(_a2), "r"(_a3), "r"(_a4), "r"(_a7) : "memory");
    return _a0;
}

int write(int fd, const void *buf, int n) {
    return (int)__syscall(SYS_write, fd, (long)buf, n);
}

int read(int fd, void *buf, int n) {
    return (int)__syscall(SYS_read, fd, (long)buf, n);
}

void yield(void) {
    (void)__syscall(SYS_yield, 0, 0, 0);
}

void exit(int status) {
    (void)__syscall(SYS_exit, status, 0, 0);
    for (;;) {
        // Should not return.
    }
}

void *sbrk(int n) {
    return (void *)__syscall(SYS_sbrk, n, 0, 0);
}

int fork(void) {
    return (int)__syscall(SYS_fork, 0, 0, 0);
}

int wait(int *status) {
    return (int)__syscall(SYS_wait, (long)status, 0, 0);
}

int exec(const char *name, char *const argv[]) {
    return (int)__syscall(SYS_exec, (long)name, (long)argv, 0);
}

int close(int fd) {
    return (int)__syscall(SYS_close, fd, 0, 0);
}

int fstat(int fd, struct stat *st) {
    return (int)__syscall(SYS_fstat, fd, (long)st, 0);
}

int chdir(const char *path) {
    return (int)__syscall(SYS_chdir, (long)path, 0, 0);
}

int dup(int fd) {
    return (int)__syscall(SYS_dup, fd, 0, 0);
}

int open(const char *name, int omode) {
    return (int)__syscall(SYS_open, (long)name, omode, 0);
}

int getpid(void) {
    return (int)__syscall(SYS_getpid, 0, 0, 0);
}

int sleep(int ticks) {
    return (int)__syscall(SYS_sleep, ticks, 0, 0);
}

int uptime(void) {
    return (int)__syscall(SYS_uptime, 0, 0, 0);
}

int kill(int pid) {
    return (int)__syscall(SYS_kill, pid, 0, 0);
}

int pipe(int fd[2]) {
    return (int)__syscall(SYS_pipe, (long)fd, 0, 0);
}

int mkdir(const char *path) {
    return (int)__syscall(SYS_mkdir, (long)path, 0, 0);
}

int unlink(const char *path) {
    return (int)__syscall(SYS_unlink, (long)path, 0, 0);
}

int link(const char *oldpath, const char *newpath) {
    return (int)__syscall(SYS_link, (long)oldpath, (long)newpath, 0);
}

int ai_call(const uint32 *tokens, int token_count, int predict_count, char *out, int out_cap) {
    return (int)__syscall5(SYS_ai_call, (long)tokens, token_count, predict_count, (long)out, out_cap);
}

int ai_submit(const uint32 *tokens, int token_count, int predict_count) {
    return (int)__syscall4(SYS_ai_submit, (long)tokens, token_count, predict_count, 0);
}

int ai_query(int reqid, struct ai_status *st) {
    return (int)__syscall(SYS_ai_query, reqid, (long)st, 0);
}

int ai_wait(int reqid, char *out, int out_cap) {
    return (int)__syscall4(SYS_ai_wait, reqid, (long)out, out_cap, 0);
}

int ai_worker_register(void) {
    return (int)__syscall(SYS_ai_worker_register, 0, 0, 0);
}

int ai_worker_get(uint32 *tokens, int token_cap, int *reqid_out, int *predict_out) {
    return (int)__syscall4(SYS_ai_worker_get, (long)tokens, token_cap, (long)reqid_out, (long)predict_out);
}

int ai_worker_complete(int reqid, const char *out, int out_len, int status) {
    return (int)__syscall4(SYS_ai_worker_complete, reqid, (long)out, out_len, status);
}
