#ifndef __PARAM_H__
#define __PARAM_H__

// Maximum number of harts (CPUs) we support.
// Keep in sync with kernel/arch/riscv/entry.S stack allocation.
#define NCPU 8

// User stack pages.
#define USERSTACK 1

// Max number of exec arguments.
#define MAXARG 32

// lazy allocation / copy-on-write 特性开关默认值。
// 通常由 Makefile 通过 -DLAZY_ALLOC=<0|1> / -DCOW_ALLOC=<0|1> 覆盖。
#ifndef LAZY_ALLOC
#define LAZY_ALLOC 1
#endif

#ifndef COW_ALLOC
#define COW_ALLOC 1
#endif

#endif
