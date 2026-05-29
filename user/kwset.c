#include "user.h"

#define KWBUF 512

static void usage(void) {
    fprintf(2, "usage: kwset FILE KEYWORD...\n");
    fprintf(2, "       kwset FILE -c    # clear keywords\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        usage();
        exit(1);
    }

    char buf[KWBUF];
    buf[0] = 0;

    if (argc == 3 && strcmp(argv[2], "-c") == 0) {
        // Empty string clears all keyword entries for this file.
    } else {
        int pos = 0;
        for (int i = 2; i < argc; i++) {
            if (i > 2) {
                if (pos + 1 >= KWBUF) {
                    fprintf(2, "kwset: keyword string too long\n");
                    exit(1);
                }
                buf[pos++] = ' ';
            }
            for (int j = 0; argv[i][j] != 0; j++) {
                if (pos + 1 >= KWBUF) {
                    fprintf(2, "kwset: keyword string too long\n");
                    exit(1);
                }
                buf[pos++] = argv[i][j];
            }
        }
        buf[pos] = 0;
    }

    if (setkeywords(argv[1], buf) < 0) {
        fprintf(2, "kwset: failed to set keywords for %s\n", argv[1]);
        exit(1);
    }

    printf("%s: %s\n", argv[1], buf[0] ? buf : "<empty>");
    exit(0);
}
