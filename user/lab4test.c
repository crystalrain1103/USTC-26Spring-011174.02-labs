#include "user.h"
#include "fcntl.h"
#include "file.h"
#include "stat.h"

#define QUERY_BUF 4096
#define HELLO_CHAIN_READ 4608
#define HELLO_TEXT_OFFSET 4096

typedef void (*test_fn)(void);

struct test_case {
    const char *name;
    test_fn fn;
};

static char qbuf[QUERY_BUF];

static void fail(const char *msg) {
    printf("  [FAIL] %s\n", msg);
    exit(1);
}

static void ok(const char *msg) {
    printf("  [PASS] %s\n", msg);
}

static int write_full(int fd, const void *buf, int n) {
    const char *p = (const char *)buf;
    int done = 0;
    while (done < n) {
        int m = write(fd, p + done, n - done);
        if (m <= 0) {
            return -1;
        }
        done += m;
    }
    return done;
}

static int read_full(int fd, void *buf, int n) {
    char *p = (char *)buf;
    int done = 0;
    while (done < n) {
        int m = read(fd, p + done, n - done);
        if (m < 0) {
            return -1;
        }
        if (m == 0) {
            break;
        }
        done += m;
    }
    return done;
}

static void fill_pattern(char *buf, int n, int seed) {
    for (int i = 0; i < n; i++) {
        buf[i] = (char)('A' + ((i + seed) % 26));
    }
}

static void create_file_with_data(const char *path, const void *data, int n) {
    int fd = open(path, O_CREATE | O_TRUNC | O_RDWR);
    if (fd < 0) {
        fail("create file failed");
    }
    if (write_full(fd, data, n) != n) {
        close(fd);
        fail("write file failed");
    }
    if (close(fd) < 0) {
        fail("close written file failed");
    }
}

static void expect_file_data(const char *path, const void *data, int n) {
    static char buf[4096];
    if (n > (int)sizeof(buf)) {
        fail("test buffer too small");
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fail("open file for readback failed");
    }
    int got = read_full(fd, buf, n);
    char extra = 0;
    int tail = read(fd, &extra, 1);
    close(fd);
    if (got != n || tail != 0) {
        fail("file length mismatch");
    }
    if (memcmp(buf, data, (uint)n) != 0) {
        fail("file data mismatch");
    }
}

static void expect_type(const char *path, short type) {
    struct stat st;
    if (stat(path, &st) < 0 || st.type != type) {
        fail("stat type mismatch");
    }
}

static int result_has_path(const char *result, const char *path) {
    int plen = (int)strlen(path);
    int i = 0;
    while (result[i] != 0) {
        int start = i;
        while (result[i] != 0 && result[i] != '\n') {
            i++;
        }
        int len = i - start;
        if (len == plen && strncmp(result + start, path, (uint)plen) == 0) {
            return 1;
        }
        if (result[i] == '\n') {
            i++;
        }
    }
    return 0;
}

static int result_line_count(const char *result) {
    int n = 0;
    for (int i = 0; result[i] != 0; i++) {
        if (result[i] == '\n') {
            n++;
        }
    }
    return n;
}

static void test_mount_and_root_stat(void) {
    expect_type("/", T_DIR);
    expect_type("/hello", T_FILE);
    ok("mount metadata and root stat");
}

static void test_root_directory_entries(void) {
    int fd = open("/", O_RDONLY);
    if (fd < 0) {
        fail("open root directory failed; check fat16_slot_by_index root branch");
    }
    int saw_hello = 0;
    for (;;) {
        struct dirent de;
        int n = read(fd, &de, sizeof(de));
        if (n < 0) {
            close(fd);
            fail("read root directory failed; check fat16_slot_by_index");
        }
        if (n == 0) {
            break;
        }
        if (n != (int)sizeof(de)) {
            close(fd);
            fail("partial root dirent; check directory read path");
        }
        if (!dirent_is_visible(&de)) {
            continue;
        }
        char name[DIRSIZ + 1];
        dirent_name_copy(&de, name, sizeof(name));
        if (strcmp(name, "hello") == 0) {
            saw_hello = 1;
        }
    }
    close(fd);
    if (!saw_hello) {
        fail("root directory misses hello; check fat16_slot_by_index root branch");
    }
    ok("root directory slot lookup");
}

static void test_packed_file_first_cluster(void) {
    int fd = open("/hello", O_RDONLY);
    if (fd < 0) {
        fail("open /hello failed; check root lookup and directory entries");
    }
    unsigned char hdr[4];
    if (read_full(fd, hdr, sizeof(hdr)) != (int)sizeof(hdr)) {
        close(fd);
        fail("short first-cluster read; check fat16_cluster_first_sector/readi");
    }
    close(fd);
    if (hdr[0] != 0x7f || hdr[1] != 'E' || hdr[2] != 'L' || hdr[3] != 'F') {
        fail("/hello header mismatch; check fat16_cluster_first_sector");
    }
    ok("packed file first-cluster read");
}

static void test_packed_file_fat_chain(void) {
    int fd = open("/hello", O_RDONLY);
    if (fd < 0) {
        fail("open /hello failed; check root lookup and directory entries");
    }
    static unsigned char buf[HELLO_CHAIN_READ];
    if (read_full(fd, buf, sizeof(buf)) != (int)sizeof(buf)) {
        close(fd);
        fail("short cross-cluster read; check fat16_read_fat/FAT chain");
    }
    close(fd);
    if (buf[0] != 0x7f || buf[1] != 'E' || buf[2] != 'L' || buf[3] != 'F') {
        fail("cross-cluster /hello header mismatch");
    }
    int nonzero_in_text_area = 0;
    for (int i = HELLO_TEXT_OFFSET; i < (int)sizeof(buf); i++) {
        if (buf[i] != 0) {
            nonzero_in_text_area = 1;
            break;
        }
    }
    if (!nonzero_in_text_area) {
        fail("cross-cluster data is empty; check fat16_read_fat");
    }
    if (buf[HELLO_TEXT_OFFSET] == 0x7f && buf[HELLO_TEXT_OFFSET + 1] == 'E' &&
        buf[HELLO_TEXT_OFFSET + 2] == 'L' && buf[HELLO_TEXT_OFFSET + 3] == 'F') {
        fail("cross-cluster read repeats first cluster; check fat16_read_fat");
    }
    ok("packed file FAT-chain read");
}

static void test_file_io(void) {
    const char *small = "lab4-small-file\n";
    create_file_with_data("/l4small", small, (int)strlen(small));
    expect_file_data("/l4small", small, (int)strlen(small));

    static char out[3000];
    fill_pattern(out, sizeof(out), 11);
    create_file_with_data("/l4big", out, sizeof(out));
    expect_file_data("/l4big", out, sizeof(out));

    const char *shorter = "xy";
    create_file_with_data("/l4small", shorter, (int)strlen(shorter));
    expect_file_data("/l4small", shorter, (int)strlen(shorter));

    ok("create, truncate, multi-sector file IO");
}

static void test_directories_and_cwd(void) {
    mkdir("/l4dir");
    expect_type("/l4dir", T_DIR);

    const char *nested = "nested-public-test";
    create_file_with_data("/l4dir/nest", nested, (int)strlen(nested));
    expect_file_data("/l4dir/nest", nested, (int)strlen(nested));

    if (chdir("/l4dir") < 0) {
        fail("chdir /l4dir failed");
    }
    char cwd[64];
    if (getcwd(cwd, sizeof(cwd)) < 0 ||
        (strcmp(cwd, "/l4dir") != 0 && strcmp(cwd, "/l4dir/") != 0)) {
        chdir("/");
        fail("getcwd after chdir mismatch");
    }
    const char *rel = "relative-public-test";
    create_file_with_data("rel", rel, (int)strlen(rel));
    chdir("/");
    expect_file_data("/l4dir/rel", rel, (int)strlen(rel));

    ok("directories, chdir, getcwd, relative IO");
}

static void test_keywords(void) {
    create_file_with_data("/l4kw", "kw", 2);
    if (setkeywords("/l4kw", "alpha beta") < 0) {
        fail("setkeywords failed");
    }
    char buf[128];
    int n = getkeywords("/l4kw", buf, sizeof(buf));
    if (n != (int)strlen("alpha beta") || strcmp(buf, "alpha beta") != 0) {
        fail("getkeywords mismatch");
    }
    if (setkeywords("/l4kw", "") < 0) {
        fail("clear keywords failed");
    }
    n = getkeywords("/l4kw", buf, sizeof(buf));
    if (n != 0 || buf[0] != 0) {
        fail("cleared keywords are not empty");
    }
    ok("set/get/clear keywords");
}

static void test_keyword_replacement_preserves_file(void) {
    const char *path = "/l4move";
    const char *data = "payload should survive keyword relocation";
    const char *long_kw = "long001 long002 long003 long004 long005 long006";
    create_file_with_data(path, data, (int)strlen(data));
    if (setkeywords(path, "short") < 0 || setkeywords(path, long_kw) < 0) {
        fail("replace with longer keywords failed");
    }
    expect_file_data(path, data, (int)strlen(data));

    char buf[160];
    int n = getkeywords(path, buf, sizeof(buf));
    if (n != (int)strlen(long_kw) || strcmp(buf, long_kw) != 0) {
        fail("long keyword replacement mismatch");
    }
    ok("keyword replacement preserves file data");
}

static void test_query_file(void) {
    create_file_with_data("/l4q1", "a", 1);
    create_file_with_data("/l4q2", "b", 1);
    create_file_with_data("/l4q3", "c", 1);

    if (setkeywords("/l4q1", "pubcommon pubalpha") < 0 ||
        setkeywords("/l4q2", "pubcommon") < 0 ||
        setkeywords("/l4q3", "pubalpha") < 0) {
        fail("set query keywords failed");
    }

    memset(qbuf, 0, sizeof(qbuf));
    int n = query_file("pubcommon pubalpha", -1, qbuf, sizeof(qbuf));
    if (n != 1 || !result_has_path(qbuf, "/l4q1") ||
        result_has_path(qbuf, "/l4q2") || result_has_path(qbuf, "/l4q3")) {
        fail("AND query result mismatch");
    }

    memset(qbuf, 0, sizeof(qbuf));
    n = query_file("pubcommon", 1, qbuf, sizeof(qbuf));
    if (n != 1 || result_line_count(qbuf) != 1) {
        fail("top_k query mismatch");
    }

    if (unlink("/l4q1") < 0) {
        fail("unlink query file failed");
    }
    memset(qbuf, 0, sizeof(qbuf));
    n = query_file("pubalpha", -1, qbuf, sizeof(qbuf));
    if (n < 0 || result_has_path(qbuf, "/l4q1")) {
        fail("deleted file still appears in query");
    }

    ok("full-scan keyword query");
}

static struct test_case tests[] = {
    { "mount metadata and root stat", test_mount_and_root_stat },
    { "root directory slot lookup", test_root_directory_entries },
    { "packed file first-cluster read", test_packed_file_first_cluster },
    { "packed file FAT-chain read", test_packed_file_fat_chain },
    { "create, truncate, multi-sector file IO", test_file_io },
    { "directories, chdir, getcwd, relative IO", test_directories_and_cwd },
    { "set/get/clear keywords", test_keywords },
    { "keyword replacement preserves file data", test_keyword_replacement_preserves_file },
    { "full-scan keyword query", test_query_file },
};

static int run_test(const struct test_case *tc) {
    int pid = fork();
    if (pid < 0) {
        printf("  [FAIL] %s: fork failed\n", tc->name);
        return 1;
    }
    if (pid == 0) {
        tc->fn();
        exit(0);
    }

    int status = 0;
    if (wait(&status) < 0) {
        printf("  [FAIL] %s: wait failed\n", tc->name);
        return 1;
    }
    if (status != 0) {
        return 1;
    }
    return 0;
}

int main(void) {
    printf("\n========================================\n");
    printf("  Lab4 Public FAT16 Test Suite\n");
    printf("  This test does not cover B+ tree bonus\n");
    printf("========================================\n\n");

    int failed = 0;
    int ntests = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < ntests; i++) {
        failed += run_test(&tests[i]);
    }

    printf("\n========================================\n");
    if (failed == 0) {
        printf("  All Lab4 public tests passed\n");
    } else {
        printf("  Lab4 public tests failed: %d/%d\n", failed, ntests);
    }
    printf("========================================\n");
    printf("LAB4_PUBLIC_DONE\n");
    exit(failed == 0 ? 0 : 1);
}
