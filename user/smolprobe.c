#define main llmrun_smol_original_main
#include "llmrun_smol.c"
#undef main

#define PAGE_SIZE_BYTES 4096ULL

static uint64 pages_for(uint64 bytes) {
    return (bytes + PAGE_SIZE_BYTES - 1ULL) / PAGE_SIZE_BYTES;
}

static void read_memstat_or_die(const char *where, struct memstat *st) {
    if (memstat(st) < 0) {
        printf("[probe] memstat failed at %s\n", where);
        exit(1);
    }
}

static int delta_used_pages(const struct memstat *a, const struct memstat *b) {
    return b->used_pages - a->used_pages;
}

static int delta_mapped_pages(const struct memstat *a, const struct memstat *b) {
    return b->proc_mapped_pages - a->proc_mapped_pages;
}

static int delta_size_pages(const struct memstat *a, const struct memstat *b) {
    return b->proc_sz_pages - a->proc_sz_pages;
}

static uint64 nonnegative_pages(int pages) {
    return pages > 0 ? (uint64)pages : 0ULL;
}

static void alloc_kv_cache_reserved(const struct model_cfg *cfg, float **kcache_out, float **vcache_out) {
    uint64 elems = (uint64)cfg->n_layers *
                   (uint64)cfg->runtime_seq_len *
                   (uint64)cfg->n_kv_heads *
                   (uint64)cfg->head_dim;

    // Reserve the maximum KV-cache virtual range, but do not touch every page.
    *kcache_out = llm_alloc_floats(elems);
    *vcache_out = llm_alloc_floats(elems);
}

static void touch_floats(float *p, uint64 count) {
    if (p != 0 && count != 0) {
        memset(p, 0, (uint)(count * sizeof(float)));
    }
}

static void touch_workspace(const struct model_cfg *cfg, struct llm_workspace *ws) {
    touch_floats(ws->hidden, (uint64)cfg->dim);
    touch_floats(ws->norm_hidden, (uint64)cfg->dim);
    touch_floats(ws->proj_out, (uint64)cfg->dim);
    touch_floats(ws->qkv_mixed, (uint64)cfg_full_q_rows(cfg));
    touch_floats(ws->query, (uint64)cfg_attn_out_dim(cfg));
    touch_floats(ws->attn_gate, (uint64)cfg_attn_out_dim(cfg));
    touch_floats(ws->key, (uint64)cfg_kv_dim(cfg));
    touch_floats(ws->value, (uint64)cfg_kv_dim(cfg));
    touch_floats(ws->attn_out, (uint64)cfg_attn_out_dim(cfg));
    touch_floats(ws->ffn_gate, (uint64)cfg->hidden_dim);
    touch_floats(ws->ffn_up, (uint64)cfg->hidden_dim);
    touch_floats(ws->scores, (uint64)cfg->runtime_seq_len);
}

int main(int argc, char *argv[]) {
    llm_set_prog_name_from_argv(argc > 0 ? argv[0] : 0, "smolprobe");
    llm_set_log_enabled(0);

    struct llm_runtime rt;
    struct llm_workspace ws;
    float *kcache = 0;
    float *vcache = 0;

    struct memstat before_kv;
    struct memstat after_reserve;
    struct memstat after_decode;

    printf("[%s] loading LLM assets...\n", llm_prog_name());
    llm_runtime_init(&rt, MODEL_KIND_SMOL, 0, argc, argv);
    llm_workspace_init_full(&rt.cfg, &ws);
    touch_workspace(&rt.cfg, &ws);

    read_memstat_or_die("before_kv", &before_kv);

    uint64 kv_elems = (uint64)rt.cfg.n_layers *
                      (uint64)rt.cfg.runtime_seq_len *
                      (uint64)rt.cfg.n_kv_heads *
                      (uint64)rt.cfg.head_dim;
    uint64 one_cache_bytes = kv_elems * sizeof(float);
    uint64 total_kv_bytes = one_cache_bytes * 2ULL;
    uint64 max_kv_pages = pages_for(total_kv_bytes);

    int total_steps = rt.prompt_n + rt.gen_n - 1;
    if (total_steps < 0) {
        total_steps = 0;
    }

    uint64 active_layer_bytes = (uint64)total_steps *
                                (uint64)rt.cfg.n_kv_heads *
                                (uint64)rt.cfg.head_dim *
                                sizeof(float);
    uint64 active_kv_pages_by_layout =
        2ULL * (uint64)rt.cfg.n_layers * pages_for(active_layer_bytes);

    printf("[%s] request prompt_tokens=%d gen_tokens=%d\n",
           llm_prog_name(), rt.prompt_n, rt.gen_n);
    printf("[%s] kv_capacity max_pages=%lu active_pages_est=%lu\n",
           llm_prog_name(), max_kv_pages, active_kv_pages_by_layout);

    alloc_kv_cache_reserved(&rt.cfg, &kcache, &vcache);

    read_memstat_or_die("after_reserve", &after_reserve);

    printf("[%s] running decode...\n", llm_prog_name());
    llm_drive_decode(&rt, &ws, kcache, vcache, 0, 0, token_forward);

    read_memstat_or_die("after_decode", &after_decode);

    int reserve_virtual = delta_size_pages(&before_kv, &after_reserve);
    int reserve_mapped = delta_mapped_pages(&before_kv, &after_reserve);
    int reserve_used = delta_used_pages(&before_kv, &after_reserve);
    int decode_mapped = delta_mapped_pages(&after_reserve, &after_decode);
    int decode_used = delta_used_pages(&after_reserve, &after_decode);
    int total_mapped = delta_mapped_pages(&before_kv, &after_decode);

    uint64 actual_mapped_pages = nonnegative_pages(total_mapped);
    uint64 saved_pages = 0;
    if (max_kv_pages > actual_mapped_pages) {
        saved_pages = max_kv_pages - actual_mapped_pages;
    }
    uint64 saved_percent = max_kv_pages == 0 ? 0 : (saved_pages * 100ULL) / max_kv_pages;

    printf("[%s] SUMMARY max_kv_pages=%lu actual_mapped_pages=%lu saved_pages=%lu saved_percent=%lu%%\n",
           llm_prog_name(), max_kv_pages, actual_mapped_pages, saved_pages, saved_percent);
    printf("[%s] reserve_phase virtual_pages=%d mapped_pages=%d used_pages=%d\n",
           llm_prog_name(), reserve_virtual, reserve_mapped, reserve_used);
    printf("[%s] decode_phase mapped_pages=%d used_pages=%d active_pages_est=%lu\n",
           llm_prog_name(), decode_mapped, decode_used, active_kv_pages_by_layout);
    if ((uint64)nonnegative_pages(reserve_mapped) >= max_kv_pages / 2ULL) {
        printf("[%s] verdict=EAGER_STYLE reserve mapped most KV pages immediately\n", llm_prog_name());
    } else {
        printf("[%s] verdict=LAZY_STYLE reserve kept KV pages unmapped; decode allocated on demand\n", llm_prog_name());
    }

    exit(0);
}
