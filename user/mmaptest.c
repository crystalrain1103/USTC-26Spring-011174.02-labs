#include "user.h"
#include "fcntl.h"
#include "mman.h"

#define PGSZ 4096
#define FILE_BYTES (PGSZ + 317)

static char file_pattern[FILE_BYTES];

static int strlen1(const char *s) {
    int n = 0;
    while (s[n] != '\0') {
        n++;
    }
    return n;
}

static void puts1(const char *s) {
    write(1, s, strlen1(s));
}

static void fail(const char *msg) {
    puts1("  [FAIL] ");
    puts1(msg);
    puts1("\n");
    exit(1);
}

static void ok(const char *msg) {
    puts1("  [PASS] ");
    puts1(msg);
    puts1("\n");
}

static void fill_pattern(char *buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = (char)('A' + (i % 23));
    }
}

static void write_full(int fd, const char *buf, int n) {
    int done = 0;
    while (done < n) {
        int w = write(fd, buf + done, n - done);
        if (w <= 0) {
            fail("write failed");
        }
        done += w;
    }
}

static void test_anon_map(void) {
    puts1("Test 1: anonymous private mmap\n");

    char *p = (char *)mmap(0, 2 * PGSZ, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        fail("anonymous mmap returned MAP_FAILED");
    }

    p[0] = 'A';
    p[PGSZ] = 'B';
    p[2 * PGSZ - 1] = 'C';
    if (p[0] != 'A' || p[PGSZ] != 'B' || p[2 * PGSZ - 1] != 'C') {
        fail("anonymous mmap data mismatch");
    }

    char *heap = (char *)sbrk(PGSZ);
    if (heap == (char *)-1) {
        fail("sbrk after mmap failed");
    }
    heap[0] = 'H';
    if (heap[0] != 'H') {
        fail("heap write after mmap failed");
    }

    if (munmap(p, 2 * PGSZ) < 0) {
        fail("anonymous munmap failed");
    }
    ok("anonymous private mmap");
}

static void prepare_file(const char *path) {
    fill_pattern(file_pattern, FILE_BYTES);
    int fd = open(path, O_CREATE | O_TRUNC | O_RDWR);
    if (fd < 0) {
        fail("open create failed");
    }
    write_full(fd, file_pattern, FILE_BYTES);
    close(fd);
}

static void verify_bytes(const char *mapped, const char *want, int n) {
    for (int i = 0; i < n; i++) {
        if (mapped[i] != want[i]) {
            fail("mapped bytes mismatch");
        }
    }
}

static void test_file_map(void) {
    const char *path = "/mmap.data";
    puts1("Test 2: file-backed private mmap\n");

    prepare_file(path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fail("open readonly failed");
    }

    char *mapped = (char *)mmap(0, FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fail("file mmap returned MAP_FAILED");
    }

    verify_bytes(mapped, file_pattern, FILE_BYTES);

    if (munmap(mapped, FILE_BYTES) < 0) {
        fail("file munmap failed");
    }
    if (unlink(path) < 0) {
        fail("unlink mapped file failed");
    }

    ok("file-backed private mmap");
}

static void test_fork_file_map(void) {
    const char *path = "/mmap.fork";
    puts1("Test 3: fork inherits mmap metadata\n");

    prepare_file(path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fail("open fork file failed");
    }

    char *mapped = (char *)mmap(0, FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fail("fork mmap returned MAP_FAILED");
    }

    int pid = fork();
    if (pid < 0) {
        fail("fork failed");
    }

    if (pid == 0) {
        verify_bytes(mapped, file_pattern, FILE_BYTES);
        if (munmap(mapped, FILE_BYTES) < 0) {
            exit(2);
        }
        exit(0);
    }

    int status = -1;
    if (wait(&status) < 0) {
        fail("wait failed");
    }
    if (status != 0) {
        fail("child failed to fault mapped pages");
    }

    if (munmap(mapped, FILE_BYTES) < 0) {
        fail("parent munmap after fork failed");
    }
    if (unlink(path) < 0) {
        fail("unlink fork file failed");
    }

    ok("fork inherits mmap metadata");
}

int main(void) {
    puts1("\n==============================\n");
    puts1("  mmap Test Suite\n");
    puts1("==============================\n\n");

    test_anon_map();
    test_file_map();
    test_fork_file_map();

    puts1("\n==============================\n");
    puts1("  All mmap tests passed\n");
    puts1("==============================\n");
    exit(0);
}
