//
// crash.c — Test that out-of-bounds access is properly caught.
// This process should be killed by the kernel (page fault on invalid address).
//

#include "user.h"

int main(void) {
    write(1, "crash: attempting out-of-bounds access...\n", 41);

    // Access memory way beyond the process's heap.
    // This should trigger a page fault with fault_va >= p->sz,
    // causing the kernel to kill this process.
    volatile char *p = (volatile char *)0xdeadbeef;
    char c = *p;
    (void)c;

    // Should not reach here
    write(1, "crash: ERROR - should have been killed!\n", 40);
    exit(1);
    return 0;
}
