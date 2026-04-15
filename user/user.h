#ifndef __USER_H__
#define __USER_H__

#include "types.h"
#include <stdarg.h>

struct stat;

int write(int fd, const void *buf, int n);
int read(int fd, void *buf, int n);
void yield(void);
void exit(int status) __attribute__((noreturn));
void *sbrk(int n);
int fork(void);
int wait(int *status);
int exec(const char *name, char *const argv[]);
int close(int fd);
int fstat(int fd, struct stat *st);
int chdir(const char *path);
int dup(int fd);
int open(const char *name, int omode);
int getpid(void);
int sleep(int ticks);
int uptime(void);
int kill(int pid);
int pipe(int fd[2]);
int mkdir(const char *path);
int unlink(const char *path);
int link(const char *oldpath, const char *newpath);
int ai_call(const uint32 *tokens, int token_count, int predict_count, char *out, int out_cap);
int ai_submit(const uint32 *tokens, int token_count, int predict_count);
int ai_query(int reqid, struct ai_status *st);
int ai_wait(int reqid, char *out, int out_cap);
int ai_worker_register(void);
int ai_worker_get(uint32 *tokens, int token_cap, int *reqid_out, int *predict_out);
int ai_worker_complete(int reqid, const char *out, int out_len, int status);

// user library helpers
int stat(const char *path, struct stat *st);
char *strcpy(char *dst, const char *src);
int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, uint n);
uint strlen(const char *s);
char *strchr(const char *s, char c);
char *gets(char *buf, int max);
int atoi(const char *s);
void *memset(void *dst, int c, uint n);
void *memmove(void *dst, const void *src, int n);
void *memcpy(void *dst, const void *src, uint n);
int memcmp(const void *a, const void *b, uint n);

// printf-style output helpers
void vprintf(int fd, const char *fmt, va_list ap);
void fprintf(int fd, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#endif
