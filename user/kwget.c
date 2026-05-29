#include "user.h"

#define KWBUF 512

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(2, "usage: kwget FILE\n");
        exit(1);
    }

    char buf[KWBUF];
    int n = getkeywords(argv[1], buf, sizeof(buf));
    if (n < 0) {
        fprintf(2, "kwget: failed to get keywords for %s\n", argv[1]);
        exit(1);
    }

    printf("%s: %s\n", argv[1], n > 0 ? buf : "<empty>");
    exit(0);
}
