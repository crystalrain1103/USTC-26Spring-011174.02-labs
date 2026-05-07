#ifndef __MEMSTAT_H__
#define __MEMSTAT_H__

struct memstat {
    int free_pages;
    int used_pages;
    int proc_mapped_pages;
    int proc_sz_pages;
};

#endif
