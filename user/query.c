#include "user.h"

#define QUERY_BUF 4096
#define QUERY_KW_BUF 512

static char result[QUERY_BUF];
static char keywords[QUERY_KW_BUF];

static void usage(void) {
    fprintf(2, "usage: query [-b] [-k TOP_K] KEYWORD...\n");
}

static int parse_int(const char *s, int *out) {
    int sign = 1;
    int i = 0;
    int n = 0;

    if (s[0] == '-') {
        sign = -1;
        i = 1;
    }
    if (s[i] == 0) {
        return -1;
    }
    for (; s[i] != 0; i++) {
        if (s[i] < '0' || s[i] > '9') {
            return -1;
        }
        n = n * 10 + s[i] - '0';
    }
    *out = sign * n;
    return 0;
}

static int build_keywords(int argc, char *argv[], int first) {
    int pos = 0;
    for (int i = first; i < argc; i++) {
        if (i > first) {
            if (pos + 1 >= QUERY_KW_BUF) {
                return -1;
            }
            keywords[pos++] = ' ';
        }
        for (int j = 0; argv[i][j] != 0; j++) {
            if (pos + 1 >= QUERY_KW_BUF) {
                return -1;
            }
            keywords[pos++] = argv[i][j];
        }
    }
    keywords[pos] = 0;
    return pos > 0 ? 0 : -1;
}

int main(int argc, char *argv[]) {
    int top_k = -1;
    int indexed = 0;
    int first_keyword = 1;

    if (argc < 2) {
        usage();
        exit(1);
    }

    while (first_keyword < argc) {
        if (strcmp(argv[first_keyword], "-b") == 0) {
            indexed = 1;
            first_keyword++;
            continue;
        }
        if (strcmp(argv[first_keyword], "-k") == 0) {
            if (first_keyword + 2 >= argc ||
                parse_int(argv[first_keyword + 1], &top_k) < 0 || top_k < -1) {
                usage();
                exit(1);
            }
            first_keyword += 2;
            continue;
        }
        break;
    }

    if (first_keyword >= argc) {
        usage();
        exit(1);
    }

    if (build_keywords(argc, argv, first_keyword) < 0) {
        fprintf(2, "query: keyword string too long\n");
        exit(1);
    }

    int n = indexed ? query_file_indexed(keywords, top_k, result, sizeof(result))
                    : query_file(keywords, top_k, result, sizeof(result));
    if (n < 0) {
        if (indexed) {
            fprintf(2, "query: indexed query failed\n");
        } else {
            fprintf(2, "query: failed\n");
        }
        exit(1);
    }

    if (result[0] != 0) {
        printf("%s", result);
    }
    exit(0);
}
