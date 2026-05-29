#include "types.h"
#include "defs.h"
#include "file.h"

#ifndef CONFIG_FS_FAT16
#define CONFIG_FS_FAT16 0
#endif

struct fs_ops {
    void (*fsinit)(int dev);
    struct inode *(*ialloc)(uint dev, short type);
    struct inode *(*namei)(const char *path);
    struct inode *(*nameiparent)(const char *path, char *name);
    struct inode *(*idup)(struct inode *ip);
    void (*ilock)(struct inode *ip);
    void (*iunlock)(struct inode *ip);
    void (*iput)(struct inode *ip);
    void (*iunlockput)(struct inode *ip);
    void (*iupdate)(struct inode *ip);
    void (*itrunc)(struct inode *ip);
    int (*readi)(struct inode *ip, uint64 off, void *dst, uint n);
    int (*writei)(struct inode *ip, uint64 off, const void *src, uint n);
    int (*namecmp)(const char *s, const char *t);
    struct inode *(*dirlookup)(struct inode *dp, const char *name, uint *poff);
    int (*dirlink)(struct inode *dp, const char *name, uint inum);
    int (*getcwd_path)(struct inode *cwd, char *buf, int max);
    void (*fs_read_file)(const char *path);
};

#if CONFIG_FS_FAT16
void fat16fs_fsinit(int dev);
struct inode *fat16fs_ialloc(uint dev, short type);
struct inode *fat16fs_namei(const char *path);
struct inode *fat16fs_nameiparent(const char *path, char *name);
struct inode *fat16fs_idup(struct inode *ip);
void fat16fs_ilock(struct inode *ip);
void fat16fs_iunlock(struct inode *ip);
void fat16fs_iput(struct inode *ip);
void fat16fs_iunlockput(struct inode *ip);
void fat16fs_iupdate(struct inode *ip);
void fat16fs_itrunc(struct inode *ip);
int fat16fs_readi(struct inode *ip, uint64 off, void *dst, uint n);
int fat16fs_writei(struct inode *ip, uint64 off, const void *src, uint n);
int fat16fs_namecmp(const char *s, const char *t);
struct inode *fat16fs_dirlookup(struct inode *dp, const char *name, uint *poff);
int fat16fs_dirlink(struct inode *dp, const char *name, uint inum);
int fat16fs_getcwd_path(struct inode *cwd, char *buf, int max);
void fat16fs_read_file(const char *path);
int fat16fs_get_keywords(const char *path, char *buf, int max);
int fat16fs_set_keywords(const char *path, const char *keywords);
int fat16fs_query_file(const char *keywords, int top_k, char *buf, int max);
int fat16fs_query_file_indexed(const char *keywords, int top_k, char *buf, int max);
#else
void legacyfs_fsinit(int dev);
struct inode *legacyfs_ialloc(uint dev, short type);
struct inode *legacyfs_namei(const char *path);
struct inode *legacyfs_nameiparent(const char *path, char *name);
struct inode *legacyfs_idup(struct inode *ip);
void legacyfs_ilock(struct inode *ip);
void legacyfs_iunlock(struct inode *ip);
void legacyfs_iput(struct inode *ip);
void legacyfs_iunlockput(struct inode *ip);
void legacyfs_iupdate(struct inode *ip);
void legacyfs_itrunc(struct inode *ip);
int legacyfs_readi(struct inode *ip, uint64 off, void *dst, uint n);
int legacyfs_writei(struct inode *ip, uint64 off, const void *src, uint n);
int legacyfs_namecmp(const char *s, const char *t);
struct inode *legacyfs_dirlookup(struct inode *dp, const char *name, uint *poff);
int legacyfs_dirlink(struct inode *dp, const char *name, uint inum);
int legacyfs_getcwd_path(struct inode *cwd, char *buf, int max);
void legacyfs_read_file(const char *path);
#endif

#if CONFIG_FS_FAT16
static const struct fs_ops fat16fs_ops = {
    .fsinit = fat16fs_fsinit,
    .ialloc = fat16fs_ialloc,
    .namei = fat16fs_namei,
    .nameiparent = fat16fs_nameiparent,
    .idup = fat16fs_idup,
    .ilock = fat16fs_ilock,
    .iunlock = fat16fs_iunlock,
    .iput = fat16fs_iput,
    .iunlockput = fat16fs_iunlockput,
    .iupdate = fat16fs_iupdate,
    .itrunc = fat16fs_itrunc,
    .readi = fat16fs_readi,
    .writei = fat16fs_writei,
    .namecmp = fat16fs_namecmp,
    .dirlookup = fat16fs_dirlookup,
    .dirlink = fat16fs_dirlink,
    .getcwd_path = fat16fs_getcwd_path,
    .fs_read_file = fat16fs_read_file,
};
#else
static const struct fs_ops legacyfs_ops = {
    .fsinit = legacyfs_fsinit,
    .ialloc = legacyfs_ialloc,
    .namei = legacyfs_namei,
    .nameiparent = legacyfs_nameiparent,
    .idup = legacyfs_idup,
    .ilock = legacyfs_ilock,
    .iunlock = legacyfs_iunlock,
    .iput = legacyfs_iput,
    .iunlockput = legacyfs_iunlockput,
    .iupdate = legacyfs_iupdate,
    .itrunc = legacyfs_itrunc,
    .readi = legacyfs_readi,
    .writei = legacyfs_writei,
    .namecmp = legacyfs_namecmp,
    .dirlookup = legacyfs_dirlookup,
    .dirlink = legacyfs_dirlink,
    .getcwd_path = legacyfs_getcwd_path,
    .fs_read_file = legacyfs_read_file,
};
#endif

static const struct fs_ops *active_fs(void) {
#if CONFIG_FS_FAT16
    return &fat16fs_ops;
#else
    return &legacyfs_ops;
#endif
}

void fsinit(int dev) { active_fs()->fsinit(dev); }
struct inode *ialloc(uint dev, short type) { return active_fs()->ialloc(dev, type); }
struct inode *namei(const char *path) { return active_fs()->namei(path); }
struct inode *nameiparent(const char *path, char *name) { return active_fs()->nameiparent(path, name); }
struct inode *idup(struct inode *ip) { return active_fs()->idup(ip); }
void ilock(struct inode *ip) { active_fs()->ilock(ip); }
void iunlock(struct inode *ip) { active_fs()->iunlock(ip); }
void iput(struct inode *ip) { active_fs()->iput(ip); }
void iunlockput(struct inode *ip) { active_fs()->iunlockput(ip); }
void iupdate(struct inode *ip) { active_fs()->iupdate(ip); }
void itrunc(struct inode *ip) { active_fs()->itrunc(ip); }
int readi(struct inode *ip, uint64 off, void *dst, uint n) { return active_fs()->readi(ip, off, dst, n); }
int writei(struct inode *ip, uint64 off, const void *src, uint n) {
    return active_fs()->writei(ip, off, src, n);
}
int namecmp(const char *s, const char *t) { return active_fs()->namecmp(s, t); }
struct inode *dirlookup(struct inode *dp, const char *name, uint *poff) {
    return active_fs()->dirlookup(dp, name, poff);
}
int dirlink(struct inode *dp, const char *name, uint inum) {
    return active_fs()->dirlink(dp, name, inum);
}
int getcwd_path(struct inode *cwd, char *buf, int max) {
    return active_fs()->getcwd_path(cwd, buf, max);
}
void fs_read_file(const char *path) { active_fs()->fs_read_file(path); }

int fs_get_keywords(const char *path, char *buf, int max) {
#if CONFIG_FS_FAT16
    return fat16fs_get_keywords(path, buf, max);
#else
    (void)path;
    if (buf && max > 0) {
        buf[0] = 0;
    }
    return -1;
#endif
}

int fs_set_keywords(const char *path, const char *keywords) {
#if CONFIG_FS_FAT16
    return fat16fs_set_keywords(path, keywords);
#else
    (void)path;
    (void)keywords;
    return -1;
#endif
}

int fs_query_file(const char *keywords, int top_k, char *buf, int max) {
#if CONFIG_FS_FAT16
    return fat16fs_query_file(keywords, top_k, buf, max);
#else
    (void)keywords;
    (void)top_k;
    if (buf && max > 0) {
        buf[0] = 0;
    }
    return -1;
#endif
}

int fs_query_file_indexed(const char *keywords, int top_k, char *buf, int max) {
#if CONFIG_FS_FAT16
    return fat16fs_query_file_indexed(keywords, top_k, buf, max);
#else
    (void)keywords;
    (void)top_k;
    if (buf && max > 0) {
        buf[0] = 0;
    }
    return -1;
#endif
}
