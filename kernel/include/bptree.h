#ifndef __BPTREE_H__
#define __BPTREE_H__

#include "types.h"

#define BPTREE_MAX_KEYS 15
#define BPTREE_VALUES_PER_CHUNK 32

struct bptree_node;

struct bptree_value_chunk {
    int n;
    void *values[BPTREE_VALUES_PER_CHUNK];
    struct bptree_value_chunk *next;
};

struct bptree_values {
    int count;
    struct bptree_value_chunk *head;
    struct bptree_value_chunk *tail;
};

typedef void *(*bptree_alloc_fn)(uint n, void *arg);

struct bptree {
    struct bptree_node *root;
    bptree_alloc_fn alloc;
    void *alloc_arg;
    int oom;
};

void bptree_init(struct bptree *tree, bptree_alloc_fn alloc, void *arg);
void bptree_reset(struct bptree *tree);
struct bptree_values *bptree_lookup(struct bptree *tree, const char *key, int len);
int bptree_insert(struct bptree *tree, const char *key, int len, void *value);
int bptree_values_contains(struct bptree_values *values, void *value);
void bptree_values_remove(struct bptree_values *values, void *value);

#endif
