#include "types.h"
#include "defs.h"
#include "bptree.h"

struct bptree_node {
    int leaf;
    int nkey;
    char *keys[BPTREE_MAX_KEYS + 1];
    struct bptree_node *child[BPTREE_MAX_KEYS + 2];
    struct bptree_values values[BPTREE_MAX_KEYS + 1];
    struct bptree_node *next;
};

void bptree_init(struct bptree *tree, bptree_alloc_fn alloc, void *arg) {
    if (tree == 0) {
        return;
    }
    tree->root = 0;
    tree->alloc = alloc;
    tree->alloc_arg = arg;
    tree->oom = 0;
}

void bptree_reset(struct bptree *tree) {
    if (tree == 0) {
        return;
    }
    tree->root = 0;
    tree->oom = 0;
}

static void *bptree_alloc(struct bptree *tree, uint n) {
    if (tree == 0 || tree->alloc == 0) {
        return 0;
    }
    void *p = tree->alloc(n, tree->alloc_arg);
    if (p == 0) {
        tree->oom = 1;
    }
    return p;
}

static int bptree_keycmp_token(const char *key, const char *token, int len) {
    int i = 0;
    while (key[i] != 0 && i < len) {
        uchar kc = (uchar)key[i];
        uchar tc = (uchar)token[i];
        if (kc != tc) {
            return (int)kc - (int)tc;
        }
        i++;
    }
    if (key[i] == 0 && i == len) {
        return 0;
    }
    return key[i] == 0 ? -1 : 1;
}

static char *bptree_strdup_key(struct bptree *tree, const char *key, int len) {
    char *s = (char *)bptree_alloc(tree, (uint)(len + 1));
    if (s == 0) {
        return 0;
    }
    memmove(s, key, (uint)len);
    s[len] = 0;
    return s;
}

int bptree_values_contains(struct bptree_values *values, void *value) {
    if (values == 0 || value == 0) {
        return 0;
    }
    for (struct bptree_value_chunk *c = values->head; c; c = c->next) {
        for (int i = 0; i < c->n; i++) {
            if (c->values[i] == value) {
                return 1;
            }
        }
    }
    return 0;
}

static int bptree_values_add(struct bptree *tree, struct bptree_values *values, void *value) {
    if (values == 0 || value == 0) {
        return -1;
    }
    if (bptree_values_contains(values, value)) {
        return 0;
    }
    if (values->tail == 0 || values->tail->n >= BPTREE_VALUES_PER_CHUNK) {
        struct bptree_value_chunk *chunk =
            (struct bptree_value_chunk *)bptree_alloc(tree, sizeof(*chunk));
        if (chunk == 0) {
            return -1;
        }
        if (values->tail) {
            values->tail->next = chunk;
        } else {
            values->head = chunk;
        }
        values->tail = chunk;
    }
    values->tail->values[values->tail->n++] = value;
    values->count++;
    return 0;
}

void bptree_values_remove(struct bptree_values *values, void *value) {
    if (values == 0 || value == 0) {
        return;
    }
    for (struct bptree_value_chunk *c = values->head; c; c = c->next) {
        for (int i = 0; i < c->n; i++) {
            if (c->values[i] == value) {
                for (int j = i + 1; j < c->n; j++) {
                    c->values[j - 1] = c->values[j];
                }
                c->n--;
                values->count--;
                return;
            }
        }
    }
}

static struct bptree_node *bptree_new_node(struct bptree *tree, int leaf) {
    struct bptree_node *node = (struct bptree_node *)bptree_alloc(tree, sizeof(*node));
    if (node == 0) {
        return 0;
    }
    node->leaf = leaf;
    return node;
}

struct bptree_values *bptree_lookup(struct bptree *tree, const char *key, int len) {
    if (tree == 0 || key == 0 || len <= 0) {
        return 0;
    }

    /*
     * LAB BONUS TODO [B.1]
     *
     * Search from tree->root down to a leaf. In each internal node, choose the
     * child whose key range may contain key[0..len). In the leaf, return the
     * posting list for an equal key, or 0 when the keyword is absent.
     *
     * Hint: bptree_keycmp_token(node->keys[i], key, len) compares a stored
     * zero-terminated key with a non-zero-terminated token slice.
     */
    return 0;
}

static int bptree_split_leaf(struct bptree *tree, struct bptree_node *node,
                             char **promoted, struct bptree_node **right) {
    (void)tree;
    (void)node;
    (void)promoted;
    (void)right;

    /*
     * LAB BONUS TODO [B.1]
     *
     * Split an overflowing leaf node. Move the upper half of keys and posting
     * lists into a new right leaf, link the leaf chain, and promote the first
     * key of the right leaf to the parent.
     */
    return -1;
}

static int bptree_split_internal(struct bptree *tree, struct bptree_node *node,
                                 char **promoted, struct bptree_node **right) {
    (void)tree;
    (void)node;
    (void)promoted;
    (void)right;

    /*
     * LAB BONUS TODO [B.1]
     *
     * Split an overflowing internal node. Promote the middle separator key and
     * move the keys/children on its right into a new internal node.
     */
    return -1;
}

static int bptree_insert_rec(struct bptree *tree, struct bptree_node *node,
                             const char *key, int len, void *value,
                             char **promoted, struct bptree_node **right) {
    (void)tree;
    (void)node;
    (void)key;
    (void)len;
    (void)value;
    (void)promoted;
    (void)right;

    /*
     * LAB BONUS TODO [B.1]
     *
     * Recursively insert value for key[0..len).
     *
     * Leaf case:
     *   - If the key already exists, append value to its posting list with
     *     bptree_values_add().
     *   - Otherwise copy the key with bptree_strdup_key(), insert a new sorted
     *     leaf entry, and create its posting list.
     *   - If the leaf overflows, call bptree_split_leaf().
     *
     * Internal case:
     *   - Descend to the correct child.
     *   - If the child split, insert the promoted separator and right child.
     *   - If this node overflows, call bptree_split_internal().
     *
     * On return, set *promoted and *right when this node split; otherwise set
     * them to 0.
     */
    return -1;
}

int bptree_insert(struct bptree *tree, const char *key, int len, void *value) {
    if (tree == 0 || len <= 0 || value == 0) {
        return -1;
    }

    /*
     * LAB BONUS TODO [B.1]
     *
     * Create the root leaf when the tree is empty. Then call bptree_insert_rec().
     * If the old root splits, allocate a new internal root containing the
     * promoted separator key and two children.
     */
    return -1;
}
