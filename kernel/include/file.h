#ifndef __FILE_H__
#define __FILE_H__

#include "types.h"
#include "sleeplock.h"

#ifndef CONFIG_FS_FAT16
#define CONFIG_FS_FAT16 0
#endif

#define NOFILE 16
#define NFILE 64
#define NINODE 64

#define ROOTINO 1
#define DIRSIZ 14
#define MAXPATH 128

#define T_DIR 1
#define T_FILE 2
#define T_DEVICE 3

#define INODE_NADDR 13

#if CONFIG_FS_FAT16
#define DIRENT_FAT16_NAME_LEN 11
#define DIRENT_FAT16_ATTR_VOLUME 0x08
#define DIRENT_FAT16_ATTR_LONG_NAME 0x0f
#define DIRENT_FAT16_NAME_FREE 0x00
#define DIRENT_FAT16_NAME_DELETED 0xe5

struct dirent {
    uchar name[11];
    uchar attr;
    uchar ntres;
    uchar crt_time_tenth;
    ushort crt_time;
    ushort crt_date;
    ushort last_acc_date;
    ushort first_cluster_hi;
    ushort wrt_time;
    ushort wrt_date;
    ushort first_cluster_lo;
    uint file_size;
} __attribute__((packed));
#else
struct dirent {
    ushort inum;
    char name[DIRSIZ];
};
#endif

static inline uchar dirent_lower_char(uchar c) {
    if (c >= 'A' && c <= 'Z') {
        return (uchar)(c - 'A' + 'a');
    }
    return c;
}

static inline int dirent_is_visible(const struct dirent *entry) {
    if (entry == 0) {
        return 0;
    }
#if CONFIG_FS_FAT16
    if (entry->name[0] == DIRENT_FAT16_NAME_FREE || entry->name[0] == DIRENT_FAT16_NAME_DELETED) {
        return 0;
    }
    if ((entry->attr & DIRENT_FAT16_ATTR_LONG_NAME) == DIRENT_FAT16_ATTR_LONG_NAME) {
        return 0;
    }
    if (entry->attr & DIRENT_FAT16_ATTR_VOLUME) {
        return 0;
    }
    return 1;
#else
    return entry->inum != 0;
#endif
}

static inline void dirent_name_copy(const struct dirent *entry, char *out, int cap) {
    if (out == 0 || cap <= 0) {
        return;
    }
    out[0] = 0;
    if (!dirent_is_visible(entry)) {
        return;
    }
#if CONFIG_FS_FAT16
    int k = 0;
    if (entry->name[0] == '.') {
        while (k < cap - 1 && k < DIRENT_FAT16_NAME_LEN && entry->name[k] != ' ') {
            out[k] = (char)entry->name[k];
            k++;
        }
        out[k] = 0;
        return;
    }
    for (int i = 0; i < 8 && k < cap - 1; i++) {
        if (entry->name[i] == ' ') {
            break;
        }
        out[k++] = (char)dirent_lower_char(entry->name[i]);
    }
    if (entry->name[8] != ' ' && k < cap - 1) {
        out[k++] = '.';
        for (int i = 8; i < DIRENT_FAT16_NAME_LEN && k < cap - 1; i++) {
            if (entry->name[i] == ' ') {
                break;
            }
            out[k++] = (char)dirent_lower_char(entry->name[i]);
        }
    }
    out[k] = 0;
#else
    int i = 0;
    while (i < DIRSIZ && i < cap - 1 && entry->name[i] != 0) {
        out[i] = entry->name[i];
        i++;
    }
    out[i] = 0;
#endif
}

enum filetype {
    FD_NONE = 0,
    FD_PIPE,
    FD_CONSOLE,
    FD_INODE,
    FD_GPU,
};

struct pipe;

struct inode {
    uint dev;
    uint inum;
    int ref;
    struct sleeplock lock;
    int valid;

    short type;
    short major;
    short minor;
    short nlink;
    uint size;
    uint addrs[INODE_NADDR];
};

struct file {
    enum filetype type;
    int ref;
    char readable;
    char writable;
    struct pipe *pipe;
    struct inode *ip;
    uint off;
};

void fileinit(void);
struct file *filealloc(void);
struct file *filedup(struct file *f);
void fileclose(struct file *f);
int filestat(struct file *f, uint64 addr);
int fileread(struct file *f, uint64 addr, int n);
int filewrite(struct file *f, uint64 addr, int n);
int fileioctl(struct file *f, int cmd, uint64 arg);

#endif
