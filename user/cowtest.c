//
// cowtest.c — Strict Copy-on-Write (COW) test suite
//
// Tests:
//   1. Basic COW: fork, parent writes, child reads original data
//   2. Child writes: child gets private copy, parent unaffected
//   3. Multiple forks: several children share pages until write
//   4. sbrk + COW: allocate with sbrk, fork, both sides write
//   5. Pipe with COW: fork, child writes buffer via pipe
//   6. Large data COW: many pages shared then selectively written
//   7. COW write under memory pressure: deep-copy fork should fail
//   8. Multiple children under memory pressure: repeated deep-copy forks fail
//   9. Proof of COW: fork succeeds when address space exceeds free memory
//  10. Kernel copyout to COW page
//  11. Kernel copyout under memory pressure
//  12. Kernel copyout across two COW pages under memory pressure
//

#include "user.h"

#define PGSIZE 4096
#define MB (1024 * 1024)

static void puts_fd(int fd, const char *s) {
    int len = 0;
    const char *p = s;
    while (*p++) len++;
    write(fd, s, len);
}

static void puts(const char *s) {
    puts_fd(1, s);
}

static void ok(const char *name) {
    puts("  [PASS] ");
    puts(name);
    puts("\n");
}

static void fail(const char *name) {
    puts("  [FAIL] ");
    puts(name);
    puts("\n");
    exit(1);
}

static int write_full(int fd, const void *buf, int n) {
    const char *p = (const char *)buf;
    int done = 0;
    while (done < n) {
        int r = write(fd, p + done, n - done);
        if (r <= 0)
            return r;
        done += r;
    }
    return done;
}

static int read_full(int fd, void *buf, int n) {
    char *p = (char *)buf;
    int done = 0;
    while (done < n) {
        int r = read(fd, p + done, n - done);
        if (r <= 0)
            return r;
        done += r;
    }
    return done;
}

static char pattern_byte(int seed, int page) {
    return (char)((seed + page * 31) & 0x7f);
}

static void fill_page_pattern(char *p, int bytes, int seed) {
    int page = 0;
    for (int off = 0; off < bytes; off += PGSIZE) {
        p[off] = pattern_byte(seed, page);
        page++;
    }
}

static void shrink_or_fail(int bytes) {
    if (sbrk(-bytes) == (char *)-1)
        fail("sbrk shrink failed");
}

// ---------- Test 1: Basic COW - child sees original data ----------
static void test_basic_cow(void) {
    puts("Test 1: Basic COW - child sees parent's original data\n");

    // Allocate a page and write a known pattern.
    char *p = sbrk(PGSIZE);
    if (p == (char *)-1) fail("sbrk failed");
    for (int i = 0; i < PGSIZE; i++)
        p[i] = (char)(i & 0x7f);

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        // Child: verify the data is intact (COW pages should be readable).
        for (int i = 0; i < PGSIZE; i++) {
            if (p[i] != (char)(i & 0x7f)) {
                puts("  [FAIL] child: data mismatch at offset\n");
                exit(1);
            }
        }
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0) fail("child failed verification");
        ok("basic COW: child reads parent data");
    }
}

// ---------- Test 2: Child writes - gets private copy ----------
static void test_child_writes(void) {
    puts("Test 2: Child writes - gets private copy, parent unaffected\n");

    char *p = sbrk(PGSIZE);
    if (p == (char *)-1) fail("sbrk failed");

    // Parent writes pattern A.
    for (int i = 0; i < PGSIZE; i++)
        p[i] = 'A';

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        // Child writes pattern B.
        for (int i = 0; i < PGSIZE; i++)
            p[i] = 'B';

        // Verify child has B everywhere.
        for (int i = 0; i < PGSIZE; i++) {
            if (p[i] != 'B') {
                puts("  [FAIL] child: expected 'B'\n");
                exit(1);
            }
        }
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0) fail("child failed");

        // After child exits, parent must still have A.
        for (int i = 0; i < PGSIZE; i++) {
            if (p[i] != 'A') fail("parent data corrupted");
        }
        ok("child write isolation");
    }
}

// ---------- Test 3: Multiple forks sharing pages ----------
static void test_multi_fork(void) {
    puts("Test 3: Multiple forks sharing pages\n");

    char *p = sbrk(PGSIZE);
    if (p == (char *)-1) fail("sbrk failed");

    p[0] = 'M';

    for (int c = 0; c < 3; c++) {
        int pid = fork();
        if (pid < 0) fail("fork failed");
        if (pid == 0) {
            // Each child writes its own value.
            p[0] = (char)('0' + c);
            if (p[0] != (char)('0' + c)) {
                puts("  [FAIL] child: wrong value\n");
                exit(1);
            }
            exit(0);
        }
    }

    // Wait for all children.
    for (int c = 0; c < 3; c++) {
        int status;
        wait(&status);
        if (status != 0) fail("a child failed");
    }

    // Parent must still have 'M'.
    if (p[0] != 'M') fail("parent data corrupted after multi-fork");
    ok("multiple forks sharing pages");
}

// ---------- Test 4: sbrk + fork + both sides write ----------
static void test_sbrk_cow(void) {
    puts("Test 4: sbrk + fork + both sides write\n");

    int npages = 4;
    char *p = sbrk(npages * PGSIZE);
    if (p == (char *)-1) fail("sbrk failed");

    for (int i = 0; i < npages; i++)
        p[i * PGSIZE] = (char)('a' + i);

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        // Child writes to all pages.
        for (int i = 0; i < npages; i++)
            p[i * PGSIZE] = (char)('A' + i);

        for (int i = 0; i < npages; i++) {
            if (p[i * PGSIZE] != (char)('A' + i)) {
                puts("  [FAIL] child: data mismatch\n");
                exit(1);
            }
        }
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0) fail("child failed");

        for (int i = 0; i < npages; i++) {
            if (p[i * PGSIZE] != (char)('a' + i))
                fail("parent data corrupted");
        }
        ok("sbrk + fork + both sides write");
    }
}

// ---------- Test 5: Pipe across COW fork ----------
static void test_pipe_cow(void) {
    puts("Test 5: Pipe with COW fork\n");

    char *buf = sbrk(PGSIZE);
    if (buf == (char *)-1) fail("sbrk failed");

    for (int i = 0; i < 64; i++)
        buf[i] = (char)(i + 1);

    int fds[2];
    if (pipe(fds) < 0) fail("pipe failed");

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        close(fds[0]);
        // Child writes the shared buffer through pipe.
        write(fds[1], buf, 64);
        close(fds[1]);
        exit(0);
    } else {
        close(fds[1]);
        char rbuf[64];
        int n = read(fds[0], rbuf, 64);
        close(fds[0]);

        int status;
        wait(&status);
        if (status != 0) fail("child failed");
        if (n != 64) fail("pipe read wrong size");

        for (int i = 0; i < 64; i++) {
            if (rbuf[i] != (char)(i + 1))
                fail("pipe data mismatch");
        }
        ok("pipe with COW fork");
    }
}

// ---------- Test 6: Large data COW with selective writes ----------
static void test_large_cow(void) {
    puts("Test 6: Large data COW with selective writes\n");

    int npages = 16;
    char *p = sbrk(npages * PGSIZE);
    if (p == (char *)-1) fail("sbrk failed");

    // Fill all pages.
    for (int i = 0; i < npages; i++)
        p[i * PGSIZE] = (char)(i + 10);

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        // Child writes only to even pages.
        for (int i = 0; i < npages; i += 2)
            p[i * PGSIZE] = (char)(i + 100);

        // Verify: even pages have new data, odd pages have original.
        for (int i = 0; i < npages; i++) {
            char expected = (i % 2 == 0) ? (char)(i + 100) : (char)(i + 10);
            if (p[i * PGSIZE] != expected) {
                puts("  [FAIL] child: large data mismatch\n");
                exit(1);
            }
        }
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0) fail("child failed");

        // Parent: all pages should still have original data.
        for (int i = 0; i < npages; i++) {
            if (p[i * PGSIZE] != (char)(i + 10))
                fail("parent large data corrupted");
        }
        ok("large data COW with selective writes");
    }
}

// ---------- Test 7: COW write under memory pressure ----------
static void test_cow_write_under_pressure(void) {
    puts("Test 7: COW write under memory pressure\n");

    int huge = 80 * MB;
    char *p = sbrk(huge);
    if (p == (char *)-1) fail("sbrk 80MB failed");

    fill_page_pattern(p, huge, 37);

    int target_page = 123;
    char before = p[target_page * PGSIZE];

    // A deep-copy fork must allocate another 80MB here and should fail on the
    // default 128MB machine. COW only shares the existing pages.
    int pid = fork();
    if (pid < 0)
        fail("fork failed under pressure; implementation looks like deep copy");

    if (pid == 0) {
        p[target_page * PGSIZE] = 'Z';
        if (p[target_page * PGSIZE] != 'Z')
            exit(1);
        if (p[(target_page + 1) * PGSIZE] != pattern_byte(37, target_page + 1))
            exit(2);
        exit(0);
    }

    int status;
    wait(&status);
    if (status != 0) fail("child failed COW write under pressure");
    if (p[target_page * PGSIZE] != before)
        fail("parent page changed after pressured COW write");

    shrink_or_fail(huge);
    ok("COW write under memory pressure");
}

// ---------- Test 8: Multiple children under memory pressure ----------
static void test_multi_fork_pressure(void) {
    puts("Test 8: Multiple children under memory pressure\n");

    int huge = 48 * MB;
    char *p = sbrk(huge);
    if (p == (char *)-1) fail("sbrk 48MB failed");

    fill_page_pattern(p, huge, 51);

    int sync[2];
    if (pipe(sync) < 0) fail("pipe failed");

    int children = 2;
    for (int i = 0; i < children; i++) {
        int pid = fork();
        if (pid < 0)
            fail("fork failed under multi-fork pressure; implementation looks like deep copy");
        if (pid == 0) {
            close(sync[1]);
            char token = 0;
            if (read_full(sync[0], &token, 1) != 1)
                exit(1);
            if (p[0] != pattern_byte(51, 0) ||
                p[17 * PGSIZE] != pattern_byte(51, 17))
                exit(2);
            close(sync[0]);
            exit(0);
        }
    }

    close(sync[0]);
    for (int i = 0; i < children; i++) {
        if (write_full(sync[1], "R", 1) != 1)
            fail("sync write failed");
    }
    close(sync[1]);

    for (int i = 0; i < children; i++) {
        int status;
        wait(&status);
        if (status != 0)
            fail("child failed under multi-fork pressure");
    }

    shrink_or_fail(huge);
    ok("multiple children can fork simultaneously under memory pressure");
}

// ---------- Test 9: Proof of COW (OOM Fork Test) ----------
static void test_cow_proof(void) {
    puts("Test 9: Proof of COW (Oversubscribe physical memory)\n");

    // QEMU 默认有 128MB 内存。我们申请并写入 80MB 数据。
    // 这将消耗掉大约 80MB 的真实物理内存。
    int huge = 80 * MB;
    char *p = sbrk(huge);
    if (p == (char *)-1) fail("sbrk 80MB failed");

    // 写入数据，强制物理页分配（如果结合了 Lazy Alloc 的话）
    for (int i = 0; i < huge; i += PGSIZE) {
        p[i] = 'X';
    }

    // 此时系统剩下不到 48MB 物理内存。
    // 如果是传统的深拷贝 fork，这里需要再分配 80MB，必定失败！
    // 只有 COW 机制下，这里才能瞬间成功。
    int pid = fork();
    if (pid < 0) {
        fail("fork failed! You are using Deep Copy, not COW!");
    }

    if (pid == 0) {
        exit(0); // 子进程直接退出
    } else {
        int status;
        wait(&status);
        if (status != 0) fail("child failed");
        
        // 测试完立刻归还内存，防止影响后续测试
        shrink_or_fail(huge);
        ok("COW proof (OOM fork successful)");
    }
}


// ---------- Test 10: Kernel write to COW page (copyout trap) ----------
static void test_copyout_cow(void) {
    puts("Test 10: Kernel write to COW page (copyout)\n");

    char *buf = sbrk(PGSIZE);
    if (buf == (char *)-1) fail("sbrk failed");
    buf[0] = 'O'; // 初始化

    int fds[2];
    if (pipe(fds) < 0) fail("pipe failed");

    int pid = fork();
    if (pid < 0) fail("fork failed");

    if (pid == 0) {
        close(fds[0]);
        // 子进程将父进程原本的数据通过 pipe 发过去
        write(fds[1], "NEW DATA", 9);
        close(fds[1]);
        exit(0);
    } else {
        close(fds[1]);
        int status;
        wait(&status); // 等子进程写完并退出
        
        // 关键点：此时 buf 是一个 COW 页面（父子共享，只读）。
        // 父进程调用 read，内核空间的 copyout 试图往只读的 buf 里写数据。
        // 如果内核代码没处理 copyout 的 COW 逻辑，这里会返回 -1 或者 panic。
        int n = read(fds[0], buf, 9);
        close(fds[0]);

        if (n != 9) {
            fail("read into COW page failed (check your copyout/walkaddr implementation!)");
        }
        
        if (buf[0] != 'N' || buf[1] != 'E') {
            fail("data not written correctly by kernel");
        }
        
        ok("kernel copyout to COW page");
    }
}

// ---------- Test 11: Kernel copyout to COW page under memory pressure ----------
static void test_copyout_cow_under_pressure(void) {
    puts("Test 11: Kernel copyout to COW page under memory pressure\n");

    static const char payload[] = "PRESSURED-COPYOUT";
    int huge = 72 * MB;
    char *p = sbrk(huge);
    if (p == (char *)-1) fail("sbrk 72MB failed");

    fill_page_pattern(p, huge, 73);
    char *buf = p + 11 * PGSIZE;

    int fds[2];
    if (pipe(fds) < 0) fail("pipe failed");

    int pid = fork();
    if (pid < 0)
        fail("fork failed under pressured copyout; implementation looks like deep copy");

    if (pid == 0) {
        close(fds[0]);
        if (write_full(fds[1], payload, sizeof(payload)) != (int)sizeof(payload))
            exit(1);
        close(fds[1]);
        exit(0);
    }

    close(fds[1]);
    if (read_full(fds[0], buf, sizeof(payload)) != (int)sizeof(payload)) {
        close(fds[0]);
        fail("read into pressured COW page failed");
    }
    close(fds[0]);

    int status;
    wait(&status);
    if (status != 0) fail("child failed before pressured copyout");
    if (memcmp(buf, payload, sizeof(payload)) != 0)
        fail("pressured copyout data mismatch");

    shrink_or_fail(huge);
    ok("kernel copyout to COW page under memory pressure");
}

// ---------- Test 12: Kernel copyout across two COW pages under pressure ----------
static void test_copyout_cow_cross_page_pressure(void) {
    puts("Test 12: Kernel copyout across two COW pages under memory pressure\n");

    static const char payload[32] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    };
    int huge = 72 * MB;
    char *p = sbrk(huge);
    if (p == (char *)-1) fail("sbrk 72MB failed");

    fill_page_pattern(p, huge, 91);
    char *dst = p + 19 * PGSIZE + (PGSIZE - 16);

    int fds[2];
    if (pipe(fds) < 0) fail("pipe failed");

    int pid = fork();
    if (pid < 0)
        fail("fork failed under cross-page copyout; implementation looks like deep copy");

    if (pid == 0) {
        close(fds[0]);
        if (write_full(fds[1], payload, sizeof(payload)) != (int)sizeof(payload))
            exit(1);
        close(fds[1]);
        exit(0);
    }

    close(fds[1]);
    if (read_full(fds[0], dst, sizeof(payload)) != (int)sizeof(payload)) {
        close(fds[0]);
        fail("cross-page read under pressure failed");
    }
    close(fds[0]);

    int status;
    wait(&status);
    if (status != 0) fail("child failed in pressured cross-page copyout");
    if (memcmp(dst, payload, sizeof(payload)) != 0)
        fail("pressured cross-page copyout mismatch");

    shrink_or_fail(huge);
    ok("kernel copyout across two COW pages under memory pressure");
}

int main(void) {
    puts("\n========================================\n");
    puts("  Copy-on-Write (COW) Test Suite\n");
    puts("========================================\n\n");

    test_basic_cow();
    test_child_writes();
    test_multi_fork();
    test_sbrk_cow();
    test_pipe_cow();
    test_large_cow();
    test_cow_write_under_pressure();
    test_multi_fork_pressure();
    test_cow_proof();
    test_copyout_cow();
    test_copyout_cow_under_pressure();
    test_copyout_cow_cross_page_pressure();

    puts("\n========================================\n");
    puts("  All COW tests passed!\n");
    puts("========================================\n");

    return 0;
}
