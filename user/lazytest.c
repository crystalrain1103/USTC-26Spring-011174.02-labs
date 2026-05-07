//
// lazytest.c — Verify Lazy Allocation correctness and compare with Eager
//
// Test cases:
//   1. Basic sbrk + write/read: sbrk a large region, write to it, read back.
//   2. Boundary test: access the last byte of allocated region.
//   3. Shrink test: sbrk with negative argument frees memory correctly.
//   4. Fork test: child inherits parent's lazy-allocated memory.
//   5. sbrk large then small: allocate big, free part, use remaining.
//   6. Page fault sequence: touch pages in non-sequential order.
//

#include "user.h"
#include "riscv.h"
#include "memlayout.h"

int main(void);

// Simple string output
static void puts(const char *s) {
    while (*s) {
        write(1, s, 1);
        s++;
    }
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

// ---------- Test 1: Proof of Laziness (Oversubscribe memory) ----------
static void test_lazy_proof(void) {
    puts("Test 1: Proof of laziness (Oversubscribe physical memory)\n");
    
    // 申请 256MB 内存。QEMU 默认只有 128MB 物理内存。
    // 如果是 Eager 分配，这里必定因为物理内存不足而返回 -1。
    int huge = 256 * 1024 * 1024; 
    char *p = sbrk(huge);
    
    if (p == (char *)-1) {
        puts("  [FAIL] Eager allocation detected! sbrk failed on 256MB request.\n");
        exit(1);
    }

    // 验证我们确实能用其中一小部分（触发几次缺页）
    p[0] = 'L';
    p[huge - 4096] = 'Z'; // 访问最后一页
    
    if (p[0] != 'L' || p[huge - 4096] != 'Z') {
        fail("Data mismatch in huge allocation");
    }

    // 测试完毕，立刻归还这 256MB 的虚拟地址空间，以免影响后续进程
    if (sbrk(-huge) == (char *)-1) {
        fail("sbrk negative failed after huge allocation");
    }

    ok("lazy proof (oversubscription)");
}

// ---------- Test 2: Basic sbrk + write/read ----------
static void test_basic(void) {
    puts("Test 2: Basic sbrk + write/read\n");
    char *p = sbrk(4096);
    if (p == (char *)-1)
        fail("sbrk returned -1");

    // Write pattern
    for (int i = 0; i < 4096; i++)
        p[i] = (char)(i & 0xff);

    // Read back
    for (int i = 0; i < 4096; i++) {
        if (p[i] != (char)(i & 0xff))
            fail("data mismatch");
    }
    ok("basic sbrk + write/read");
}

// ---------- Test 3: Large allocation (multi-page) ----------
static void test_large(void) {
    puts("Test 3: Large allocation (multi-page)\n");
    int npages = 10;
    int sz = npages * 4096;
    char *p = sbrk(sz);
    if (p == (char *)-1)
        fail("sbrk returned -1");

    // Write to first byte of each page
    for (int i = 0; i < npages; i++) {
        p[i * 4096] = (char)(i + 1);
    }

    // Read back
    for (int i = 0; i < npages; i++) {
        if (p[i * 4096] != (char)(i + 1))
            fail("large alloc data mismatch");
    }
    ok("large multi-page allocation");
}

// ---------- Test 4: Non-sequential page access ----------
static void test_nonseq(void) {
    puts("Test 4: Non-sequential page access\n");
    int npages = 8;
    int sz = npages * 4096;
    char *p = sbrk(sz);
    if (p == (char *)-1)
        fail("sbrk returned -1");

    // Touch pages in reverse order to trigger page faults non-sequentially
    for (int i = npages - 1; i >= 0; i--) {
        p[i * 4096] = (char)(i * 3 + 7);
    }

    for (int i = 0; i < npages; i++) {
        if (p[i * 4096] != (char)(i * 3 + 7))
            fail("non-sequential data mismatch");
    }
    ok("non-sequential page access");
}

// ---------- Test 5: Shrink with negative sbrk ----------
static void test_shrink(void) {
    puts("Test 5: Shrink with negative sbrk\n");
    char *p = sbrk(8192);  // allocate 2 pages
    if (p == (char *)-1)
        fail("sbrk returned -1");

    p[0] = 'A';
    p[4096] = 'B';

    // Shrink by 1 page
    char *old = sbrk(-4096);
    if (old == (char *)-1)
        fail("sbrk shrink returned -1");

    // First page should still be accessible
    if (p[0] != 'A')
        fail("first page data lost after shrink");

    ok("shrink with negative sbrk");
}

// ---------- Test 6: Fork inherits lazy pages ----------
static void test_fork(void) {
    puts("Test 6: Fork inherits lazy-allocated memory\n");
    char *p = sbrk(4096);
    if (p == (char *)-1)
        fail("sbrk returned -1");

    // Write before fork
    p[0] = 'X';
    p[100] = 'Y';

    int pid = fork();
    if (pid < 0)
        fail("fork failed");

    if (pid == 0) {
        // Child: verify inherited data
        if (p[0] != 'X' || p[100] != 'Y') {
            puts("  [FAIL] child: inherited data mismatch\n");
            exit(1);
        }
        // Child: allocate more and write
        char *cp = sbrk(4096);
        if (cp == (char *)-1) {
            puts("  [FAIL] child sbrk failed\n");
            exit(1);
        }
        cp[0] = 'Z';
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0)
            fail("child exited with error");
        ok("fork inherits lazy pages");
    }
}

// ---------- Test 7: sbrk(0) returns current break ----------
static void test_sbrk_zero(void) {
    puts("Test 7: sbrk(0) returns current break\n");
    char *a = sbrk(0);
    char *b = sbrk(4096);
    char *c = sbrk(0);

    if (a != b)
        fail("sbrk(0) before != sbrk(n)");
    if (c != b + 4096)
        fail("sbrk(0) after is wrong");
    ok("sbrk(0) returns current break");
}

static void test_guard_page_fault(void) {
    puts("Test 8: Guard page fault kills child without panicking kernel\n");
    int pid = fork();
    if (pid < 0)
        fail("fork failed");

    if (pid == 0) {
        volatile char stack_byte = 0;
        char *guard = (char *)(PGROUNDDOWN((uint64)&stack_byte) - 8);
        *guard = 'G';
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status == 0)
            fail("guard page write unexpectedly succeeded");
        ok("guard page fault handled as kill");
    }
}

static void test_text_write_fault(void) {
    puts("Test 9: Text write fault kills child without panicking kernel\n");
    int pid = fork();
    if (pid < 0)
        fail("fork failed");

    if (pid == 0) {
        volatile char *text = (char *)(uint64)main;
        *text = 0;
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status == 0)
            fail("text write unexpectedly succeeded");
        ok("text write protection fault handled as kill");
    }
}

static void test_lazy_sbrk_limit(void) {
    puts("Test 10: Lazy sbrk stops at TRAPFRAME boundary\n");
    int pid = fork();
    if (pid < 0)
        fail("fork failed");

    if (pid == 0) {
        const int chunk = 1 << 30;
        uint64 cur = (uint64)sbrk(0);
        while (cur + (uint64)chunk < TRAPFRAME) {
            if (sbrk(chunk) == (char *)-1)
                exit(2);
            cur += (uint64)chunk;
        }

        int final = (int)(TRAPFRAME - cur);
        if (final > 0 && sbrk(final) == (char *)-1)
            exit(3);
        if (sbrk(PGSIZE) != (char *)-1)
            exit(4);
        exit(0);
    } else {
        int status;
        wait(&status);
        if (status != 0)
            fail("lazy sbrk boundary check failed");
        ok("lazy sbrk rejects growth past TRAPFRAME");
    }
}



int main(void) {
    puts("\n========================================\n");
    puts("  Lazy Allocation Test Suite\n");
    puts("========================================\n\n");

    test_lazy_proof();
    test_basic();
    test_large();
    test_nonseq();
    test_shrink();
    test_fork();
    test_sbrk_zero();
    test_guard_page_fault();
    test_text_write_fault();
    test_lazy_sbrk_limit();
    

    puts("\n========================================\n");
    puts("  All tests passed!\n");
    puts("========================================\n");

    return 0;
}
