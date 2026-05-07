#ifndef __MMAN_H__
#define __MMAN_H__

#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4

#define MAP_PRIVATE 0x01
#define MAP_ANON    0x02

#define MAP_FAILED ((void *)-1L)

#endif
