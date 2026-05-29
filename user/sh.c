#include "user.h"
#include "fcntl.h"

#define MAXLINE 256
#define MAXARGS 16
#define MAXCMDS 32
#define STDIN_FILENO 0
#define STDOUT_FILENO 1

static int isspace1(char c) {
    return c == ' ' || c == '\t';
}

static char *skipspace(char *s) {
    while (*s && isspace1(*s)) {
        s++;
    }
    return s;
}

static void rstrip(char *s) {
    int n = (int)strlen(s);
    while (n > 0 && isspace1(s[n - 1])) {
        s[n - 1] = '\0';
        n--;
    }
}

static int readline(char *buf, int cap) {
    int i = 0;
    while (i + 1 < cap) {
        char c = 0;
        int n = read(0, &c, 1);
        if (n <= 0) {
            return -1;
        }
        if (c == '\r') {
            c = '\n';
        }
        if (c == '\n') {
            buf[i] = '\0';
            return 0;
        }
        buf[i++] = c;
    }
    buf[i] = '\0';
    return 0;
}

static int makepath(const char *cmd, char *path, int cap) {
    int i = 0;
    if (cmd[0] != '/') {
        if (cap < 2) {
            return -1;
        }
        path[i++] = '/';
    }
    for (int j = 0; cmd[j] != '\0'; j++) {
        if (i + 1 >= cap) {
            return -1;
        }
        path[i++] = cmd[j];
    }
    path[i] = '\0';
    return 0;
}

/* Split line by ';', fill cmds[], return count. Modifies line in place. */
static int split_commands(char *line, char *cmds[], int maxcmds) {
    int n = 0;
    char *p = line;
    while (*p && n < maxcmds) {
        while (*p == ';' || isspace1(*p)) {
            p++;
        }
        if (*p == '\0') {
            break;
        }
        cmds[n++] = p;
        while (*p && *p != ';') {
            p++;
        }
        if (*p == ';') {
            *p = '\0';
            p++;
        }
    }
    cmds[n] = 0;
    return n;
}

static int parseargs(char *line, char *argv[], int maxargs) {
    int argc = 0;
    char *s = skipspace(line);
    rstrip(s);

    while (*s) {
        if (argc + 1 >= maxargs) {
            return -1;
        }
        argv[argc++] = s;
        while (*s && !isspace1(*s)) {
            s++;
        }
        if (*s == '\0') {
            break;
        }
        *s = '\0';
        s = skipspace(s + 1);
    }
    argv[argc] = 0;
    return argc;
}

/* Process redirects: remove >, >>, < and their args from argv, set fd[0] and fd[1]. */
static int process_redirect(int argc, char *argv[], int fd[2]) {
    fd[0] = STDIN_FILENO;
    fd[1] = STDOUT_FILENO;
    int j = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], ">") == 0) {
            if (i + 1 >= argc) {
                break;
            }
            int tfd = open(argv[i + 1], O_RDWR | O_CREATE | O_TRUNC);
            if (tfd >= 0) {
                fd[1] = tfd;
            } else {
                printf("sh: open '%s' error\n", argv[i + 1]);
                return -1;
            }
            i++;
        } else if (strcmp(argv[i], ">>") == 0) {
            if (i + 1 >= argc) {
                break;
            }
            int tfd = open(argv[i + 1], O_RDWR | O_CREATE | O_APPEND);
            if (tfd >= 0) {
                fd[1] = tfd;
            } else {
                printf("sh: open '%s' error\n", argv[i + 1]);
                return -1;
            }
            i++;
        } else if (strcmp(argv[i], "<") == 0) {
            if (i + 1 >= argc) {
                break;
            }
            int tfd = open(argv[i + 1], O_RDONLY);
            if (tfd >= 0) {
                fd[0] = tfd;
            } else {
                printf("sh: open '%s' error\n", argv[i + 1]);
                return -1;
            }
            i++;
        } else {
            argv[j++] = argv[i];
        }
    }
    argv[j] = 0;
    return j;
}

static int run_builtin(int argc, char *argv[]) {
    if (argc == 0) {
        return 0;
    }
    if (strcmp(argv[0], "help") == 0) {
        printf("Run program by name with args, e.g. sleep 20, echo hi\n");
        printf("Builtins: cd [dir], exit. Use ; for multiple commands.\n");
        printf("Redirect: >, >>, <\n");
        return 0;
    }
    if (strcmp(argv[0], "cd") == 0) {
        const char *target = "/";
        if (argc > 2) {
            printf("[sh] usage: cd [dir]\n");
            return 0;
        }
        if (argc == 2) {
            target = argv[1];
        }
        if (chdir(target) < 0) {
            printf("[sh] cd failed\n");
        }
        return 0;
    }
    if (strcmp(argv[0], "exit") == 0) {
        exit(0);
    }
    return -1;
}

int main(void) {
    char line[MAXLINE];
    char path[MAXLINE];
    char *argv[MAXARGS];
    char *cmds[MAXCMDS];

    for (;;) {
        char cwd[128];
        if (getcwd(cwd, sizeof(cwd)) < 0) {
            cwd[0] = '/';
            cwd[1] = '\0';
        }
        printf("shell:%s -> ", cwd);

        if (readline(line, sizeof(line)) < 0) {
            printf("\n[sh] input closed\n");
            exit(0);
        }

        int ncmds = split_commands(line, cmds, MAXCMDS);

        for (int ci = 0; ci < ncmds; ci++) {
            rstrip(skipspace(cmds[ci]));
            if (cmds[ci][0] == '\0') {
                continue;
            }

            int argc = parseargs(cmds[ci], argv, MAXARGS);
            if (argc < 0) {
                printf("[sh] too many args\n");
                continue;
            }
            if (argc == 0) {
                continue;
            }

            int fd[2];
            argc = process_redirect(argc, argv, fd);
            if (argc < 0) {
                continue;
            }

            if (run_builtin(argc, argv) == 0) {
                if (fd[0] != STDIN_FILENO) {
                    close(fd[0]);
                }
                if (fd[1] != STDOUT_FILENO) {
                    close(fd[1]);
                }
                continue;
            }

            char *cmd = argv[0];
            if (makepath(cmd, path, sizeof(path)) < 0) {
                printf("[sh] command too long\n");
                if (fd[0] != STDIN_FILENO) close(fd[0]);
                if (fd[1] != STDOUT_FILENO) close(fd[1]);
                continue;
            }
            argv[0] = path;

            int pid = fork();
            if (pid < 0) {
                printf("[sh] fork failed\n");
                if (fd[0] != STDIN_FILENO) close(fd[0]);
                if (fd[1] != STDOUT_FILENO) close(fd[1]);
                continue;
            }
            if (pid == 0) {
                dup2(fd[0], STDIN_FILENO);
                dup2(fd[1], STDOUT_FILENO);
                if (fd[0] != STDIN_FILENO) close(fd[0]);
                if (fd[1] != STDOUT_FILENO) close(fd[1]);
                exec(path, argv);
                printf("[sh] exec failed\n");
                exit(127);
            }

            if (fd[0] != STDIN_FILENO) close(fd[0]);
            if (fd[1] != STDOUT_FILENO) close(fd[1]);

            int status = 0;
            if (wait(&status) < 0) {
                printf("[sh] wait failed\n");
            }
        }
    }
}
