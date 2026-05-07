#ifndef __DEFS_H__
#define __DEFS_H__

#include "types.h"
#include "riscv.h"

struct proc;

// drivers/uart.c
void uart_init(void);
void uart_putc(char ch);
void uart_puts(char *s);
int uart_getc(void);
void uart_isr(void);

// drivers/plic.c
void plic_init(void);
void plic_inithart(void);
int plic_claim(void);
void plic_complete(int irq);

// core/console.c
void consoleinit(void);
void consolepanic(void);
void consoleacquire(void);
void consolerelease(void);
void consputc(int c);
void consoleintr(int c);
int consolegetc(void);
int consoleread(pagetable_t pagetable, uint64 dstva, int n);

// core/file.c
struct file;
void fileinit(void);
struct file *filealloc(void);
struct file *filedup(struct file *f);
void fileclose(struct file *f);
int filestat(struct file *f, uint64 addr);
int fileread(struct file *f, uint64 addr, int n);
int filewrite(struct file *f, uint64 addr, int n);

// core/pipe.c
struct pipe;
int pipealloc(struct file **f0, struct file **f1);
void pipeclose(struct pipe *pi, int writable);
int pipewrite(struct pipe *pi, uint64 addr, int n);
int piperead(struct pipe *pi, uint64 addr, int n);

// fs/fs.c
struct inode;
void fsinit(int dev);
struct inode *ialloc(uint dev, short type);
struct inode *namei(const char *path);
struct inode *nameiparent(const char *path, char *name);
struct inode *idup(struct inode *ip);
void ilock(struct inode *ip);
void iunlock(struct inode *ip);
void iput(struct inode *ip);
void iunlockput(struct inode *ip);
void iupdate(struct inode *ip);
void itrunc(struct inode *ip);
int readi(struct inode *ip, uint64 off, void *dst, uint n);
int readi_user(struct inode *ip, uint64 off, pagetable_t pagetable, uint64 dstva, uint n);
int writei(struct inode *ip, uint64 off, const void *src, uint n);
int namecmp(const char *s, const char *t);
struct inode *dirlookup(struct inode *dp, const char *name, uint *poff);
int dirlink(struct inode *dp, const char *name, uint inum);
void fs_read_file(const char *path);

// lib/printf.c
void printfinit(void);
void printf(char *fmt, ...);
void panic(char *s);

// lib/string.c
void *memset(void *dst, int c, uint n);
void *memmove(void *dst, const void *src, uint n);

// core/kalloc.c
void kinit(void);
void kfree(void *pa);
void *kalloc(void);
int kfreepage_count(void);
int ktotalpage_count(void);

// core/vm.c
void kvminit(void);
void kvminithart(void);
extern pagetable_t kernel_pagetable;
pte_t *walk(pagetable_t pagetable, uint64 va, int alloc);
int mappages(pagetable_t pagetable, uint64 va, uint64 size, uint64 pa, int perm);
pagetable_t uvmcreate(void);
uint64 walkaddr(pagetable_t pagetable, uint64 va);
void uvmunmap(pagetable_t pagetable, uint64 va, uint64 npages, int do_free);
uint64 uvmalloc(pagetable_t pagetable, uint64 oldsz, uint64 newsz, int xperm);
uint64 uvmdealloc(pagetable_t pagetable, uint64 oldsz, uint64 newsz);
int uvmmapped_pages(pagetable_t pagetable, uint64 sz);
int uvmcopy(pagetable_t old, pagetable_t new, uint64 sz);
int uvmcopyrange(pagetable_t old, pagetable_t new, uint64 start, uint64 len);
// Lazy allocation / COW: expose page-fault helpers used by trap and
// copyin/copyout paths.
int cow_handle_fault(pagetable_t pagetable, uint64 va);
int user_lazy_alloc(struct proc *p, pagetable_t pagetable, uint64 va);
// COW: expose refcount helpers used when sharing and releasing pages.
void kaddref(void *pa);
int kgetref(void *pa);

void uvmfree(pagetable_t pagetable, uint64 sz);
void uvmclear(pagetable_t pagetable, uint64 va);
int copyout(pagetable_t pagetable, uint64 dstva, char *src, uint64 len);
int copyin(pagetable_t pagetable, char *dst, uint64 srcva, uint64 len);
int copyinstr(pagetable_t pagetable, char *dst, uint64 srcva, uint64 max);

// core/syscall.c
void syscall(void);

// core/ai_service.c
void ai_service_init(void);
int ai_service_submit(uint64 prompt_uva, int prompt_len);
int ai_service_query(int reqid, uint64 st_uva);
int ai_service_wait(int reqid, uint64 out_uva, int out_cap);
int ai_service_call(uint64 prompt_uva, int prompt_len, uint64 out_uva, int out_cap);

// core/proc.c
struct spinlock;
struct proc;
int kill(int pid);
struct proc *myproc(void);
struct inode *proc_cwddup(void);
uint64 proc_mmap_limit(struct proc *p);
uint64 proc_mmap(uint64 addr, uint64 len, int prot, int flags, struct file *f, uint64 off);
int proc_munmap(uint64 addr, uint64 len);
int proc_handle_page_fault(uint64 fault_va, int write);
void proc_free_mmaps(struct proc *p, pagetable_t pagetable);
int proc_copy_mmaps(struct proc *dst, struct proc *src);
int proc_spawn_user_service(const char *path);
int proc_spawn_user_service_argv(const char *path, char *const argv[]);
int proc_wait_pid(int pid, int *status_out);
int unlink(const char *path);
void sleep(void *chan, struct spinlock *lk);
void wakeup(void *chan);

// core/sleeplock.c
struct sleeplock;
void initsleeplock(struct sleeplock *lk, char *name);
void acquiresleep(struct sleeplock *lk);
void releasesleep(struct sleeplock *lk);
int holdingsleep(struct sleeplock *lk);

// core/trap.c
void usertrap(void);
void usertrapret(void);
uint ticks_get(void);
int ticks_sleep(uint n);

// drivers/virtio_disk.c
struct buf;
void virtio_disk_init(void);
void virtio_disk_rw(struct buf *b, int write);
void virtio_disk_isr(void);

// core/bio.c
void binit(void);
struct buf *bread(uint dev, uint blockno);
void bwrite(struct buf *b);
void brelse(struct buf *b);

#endif
