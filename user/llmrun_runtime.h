#ifndef __LLMRUN_RUNTIME_H__
#define __LLMRUN_RUNTIME_H__

#include "llmrun_support.h"

struct llm_runtime {
    struct model_cfg cfg;
    struct qmat emb;
    uint32 *layer_kinds;
    float *final_norm;
    struct layer *layers;
    uint32 *prompt;
    int prompt_n;
    int gen_n;
    uint32 *expect;
    int expect_n;
    float *rope;
    uint32 *seq;
};

struct llm_request_override {
    int enabled;
    uint32 *prompt;
    int prompt_n;
    int gen_n;
};

typedef int (*llm_token_forward_fn)(
    const struct llm_runtime *rt,
    float *kcache,
    float *vcache,
    float *linear_conv_cache,
    float *linear_state_cache,
    int pos,
    struct llm_workspace *ws
);

static const char *llm_model_name(uint32 kind) {
    if (kind == MODEL_KIND_SMOL) {
        return "SmolLM";
    }
    if (kind == MODEL_KIND_QWEN) {
        return "Qwen3.5";
    }
    return "unknown-model";
}

static float *llm_alloc_floats(uint64 count) {
    if (count == 0) {
        return 0;
    }
    return (float *)xmalloc(count * sizeof(float));
}

static float *llm_alloc_zeroed_floats(uint64 count) {
    float *buf = llm_alloc_floats(count);
    if (buf != 0) {
        memset(buf, 0, (uint)(count * sizeof(float)));
    }
    return buf;
}

static float *llm_kcache_at(float *cache, const struct model_cfg *cfg, int layer, int pos, int kv_head) {
    uint64 idx = (((uint64)layer * (uint64)cfg->runtime_seq_len + (uint64)pos) * (uint64)cfg->n_kv_heads + (uint64)kv_head) *
                 (uint64)cfg->head_dim;
    return cache + idx;
}

static uint64 llm_elapsed_us(uint64 start_raw, uint64 end_raw) {
    return llm_raw_time_to_us(end_raw - start_raw);
}

static uint64 llm_avg_us(uint64 total_us, int count) {
    if (count <= 0) {
        return 0;
    }
    return (total_us + (uint64)count / 2ULL) / (uint64)count;
}

static uint64 llm_tokens_per_s_x100(int tokens, uint64 total_us) {
    if (tokens <= 0 || total_us == 0) {
        return 0;
    }
    return (((uint64)tokens * 100000000ULL) + total_us / 2ULL) / total_us;
}

static void llm_log_ms_metric(const char *name, uint64 us, const char *scope) {
    if (!llm_log_enabled()) {
        return;
    }
    LLM_LOG("%s_ms=", name);
    llm_print_ms_value(us);
    printf(" scope=%s\n", scope);
}

static void llm_log_rate_metric(const char *name, uint64 x100, const char *scope) {
    if (!llm_log_enabled()) {
        return;
    }
    LLM_LOG("%s_tok_per_s=", name);
    llm_print_rate_value(x100);
    printf(" scope=%s\n", scope);
}

static int llm_parse_positive_int_arg(const char *s, int *out) {
    uint64 v = 0;

    if (s == 0 || *s == '\0') {
        return -1;
    }
    while (*s != '\0') {
        if (*s < '0' || *s > '9') {
            return -1;
        }
        v = v * 10ULL + (uint64)(*s - '0');
        if (v > 0x7fffffffULL) {
            return -1;
        }
        s++;
    }
    if (v == 0) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int llm_parse_token_csv_arg(const char *s, uint32 **tokens_out, int *count_out) {
    int count = 1;
    const char *p = s;

    if (s == 0 || *s == '\0') {
        return -1;
    }
    while (*p != '\0') {
        if (*p == ',') {
            if (p == s || p[1] == '\0' || p[1] == ',') {
                return -1;
            }
            count++;
        } else if (*p < '0' || *p > '9') {
            return -1;
        }
        p++;
    }

    uint32 *tokens = (uint32 *)xmalloc(sizeof(uint32) * (uint64)count);
    p = s;
    for (int i = 0; i < count; i++) {
        uint64 v = 0;
        int saw_digit = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10ULL + (uint64)(*p - '0');
            if (v > 0xffffffffULL) {
                return -1;
            }
            p++;
            saw_digit = 1;
        }
        if (!saw_digit) {
            return -1;
        }
        tokens[i] = (uint32)v;
        if (i + 1 < count) {
            if (*p != ',') {
                return -1;
            }
            p++;
        } else if (*p != '\0') {
            return -1;
        }
    }

    *tokens_out = tokens;
    *count_out = count;
    return 0;
}

static void llm_parse_request_override(int argc, char *argv[], struct llm_request_override *override) {
    char *token_csv = 0;
    char *gen_arg = 0;

    memset(override, 0, sizeof(*override));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt-tokens") == 0) {
            if (token_csv != 0 || i + 1 >= argc) {
                LLM_ERR("bad --prompt-tokens usage\n");
                exit(1);
            }
            token_csv = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--gen-len") == 0) {
            if (gen_arg != 0 || i + 1 >= argc) {
                LLM_ERR("bad --gen-len usage\n");
                exit(1);
            }
            gen_arg = argv[++i];
            continue;
        }
        LLM_ERR("unknown arg %s\n", argv[i]);
        exit(1);
    }

    if (token_csv == 0 && gen_arg == 0) {
        return;
    }
    if (token_csv == 0 || gen_arg == 0) {
        LLM_ERR("custom decode requires both --prompt-tokens and --gen-len\n");
        exit(1);
    }
    if (llm_parse_token_csv_arg(token_csv, &override->prompt, &override->prompt_n) < 0) {
        LLM_ERR("bad --prompt-tokens payload\n");
        exit(1);
    }
    if (llm_parse_positive_int_arg(gen_arg, &override->gen_n) < 0) {
        LLM_ERR("bad --gen-len value\n");
        exit(1);
    }
    override->enabled = 1;
}

static void llm_runtime_install_request(
    struct llm_runtime *rt,
    uint32 *prompt,
    int prompt_n,
    uint32 *expect,
    int expect_n,
    int gen_n
) {
    if (prompt == 0 || prompt_n <= 0 || gen_n <= 0 || expect_n < 0 || expect_n > gen_n) {
        LLM_ERR("invalid request shape\n");
        exit(1);
    }
    if (prompt_n + gen_n > rt->cfg.runtime_seq_len) {
        LLM_ERR("request exceeds runtime sequence length\n");
        exit(1);
    }
    for (int i = 0; i < prompt_n; i++) {
        if (prompt[i] >= (uint32)rt->cfg.vocab_size) {
            LLM_ERR("prompt token %d out of range: %u\n", i, prompt[i]);
            exit(1);
        }
    }
    for (int i = 0; i < expect_n; i++) {
        if (expect[i] >= (uint32)rt->cfg.vocab_size) {
            LLM_ERR("expect token %d out of range: %u\n", i, expect[i]);
            exit(1);
        }
    }

    rt->prompt = prompt;
    rt->prompt_n = prompt_n;
    rt->gen_n = gen_n;
    rt->expect = expect;
    rt->expect_n = expect_n;
    rt->seq = (uint32 *)xmalloc(sizeof(uint32) * (uint64)rt->cfg.runtime_seq_len);
    memset(rt->seq, 0, (uint)(sizeof(uint32) * (uint64)rt->cfg.runtime_seq_len));
    for (int i = 0; i < rt->prompt_n; i++) {
        rt->seq[i] = rt->prompt[i];
    }
    LLM_LOG("prompt=%d gen=%d expect=%d\n", rt->prompt_n, rt->gen_n, rt->expect_n);
}

static void llm_runtime_init(struct llm_runtime *rt, uint32 required_model_kind, int allow_linear, int argc, char *argv[]) {
    memset(rt, 0, sizeof(*rt));
    uint64 load_start_raw = llm_now_raw_time();

    LLM_LOG("loading LLM assets\n");
    load_model_cfg(&rt->cfg);
    if (rt->cfg.model_kind != required_model_kind) {
        LLM_ERR("expected %s assets, got %s\n",
                llm_model_name(required_model_kind), llm_model_name(rt->cfg.model_kind));
        exit(1);
    }
    LLM_LOG("cfg kind=%d dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d runtime_seq=%d\n",
            (int)rt->cfg.model_kind, rt->cfg.dim, rt->cfg.hidden_dim, rt->cfg.n_layers, rt->cfg.n_heads,
            rt->cfg.n_kv_heads, rt->cfg.head_dim, rt->cfg.vocab_size, rt->cfg.runtime_seq_len);

    rt->layer_kinds = load_layer_kinds(&rt->cfg);
    int linear_layers = 0;
    for (int l = 0; l < rt->cfg.n_layers; l++) {
        if (rt->layer_kinds[l] == LAYER_KIND_LINEAR) {
            linear_layers++;
        }
    }
    if (!allow_linear && linear_layers != 0) {
        LLM_ERR("this program only supports full-attention layers\n");
        exit(1);
    }
    if (linear_layers != 0 && cfg_linear_conv_dim(&rt->cfg) <= 0) {
        LLM_ERR("linear layer metadata is incomplete\n");
        exit(1);
    }

    uint64 emb_blob_sz = 0;
    char *emb_blob = load_file("EMB.BIN", &emb_blob_sz);
    if (emb_blob == 0) {
        LLM_ERR("missing EMB.BIN\n");
        exit(1);
    }
    LLM_LOG("EMB.BIN loaded (%lu bytes)\n", (unsigned long)emb_blob_sz);
    rt->emb.rows = rt->cfg.vocab_size;
    rt->emb.cols = rt->cfg.dim;
    rt->emb.scales = (float *)emb_blob;
    rt->emb.q = (int8 *)(emb_blob + (uint64)rt->cfg.vocab_size * sizeof(float));
    if ((uint64)((char *)rt->emb.q - emb_blob) + (uint64)rt->cfg.vocab_size * (uint64)rt->cfg.dim != emb_blob_sz) {
        LLM_ERR("embedding blob size mismatch\n");
        exit(1);
    }

    uint64 nrm_sz = 0;
    rt->final_norm = (float *)load_file("NRM.BIN", &nrm_sz);
    if (nrm_sz != (uint64)rt->cfg.dim * sizeof(float)) {
        LLM_ERR("bad NRM.BIN size\n");
        exit(1);
    }

    rt->layers = (struct layer *)xmalloc(sizeof(struct layer) * (uint64)rt->cfg.n_layers);
    for (int l = 0; l < rt->cfg.n_layers; l++) {
        char path[16];
        uint64 sz = 0;
        char *blob;

        make_layer_path(l, path, sizeof(path));
        blob = load_file(path, &sz);
        if (blob == 0) {
            LLM_ERR("missing %s\n", path);
            exit(1);
        }
        parse_layer_blob(&rt->layers[l], blob, sz, &rt->cfg, rt->layer_kinds[l]);
        if ((l % 5) == 4 || l == rt->cfg.n_layers - 1) {
            LLM_LOG("layers 0..%d loaded\n", l);
        }
    }
    LLM_LOG("layer mix full=%d linear=%d\n", rt->cfg.n_layers - linear_layers, linear_layers);

    uint64 rope_sz = 0;
    rt->rope = (float *)load_file("ROP.BIN", &rope_sz);
    if (rope_sz != (uint64)rt->cfg.runtime_seq_len * (uint64)rt->cfg.rope_rotary_dim * sizeof(float)) {
        LLM_ERR("bad ROP.BIN size\n");
        exit(1);
    }

    struct llm_request_override override;
    llm_parse_request_override(argc, argv, &override);
    if (override.enabled) {
        llm_runtime_install_request(rt, override.prompt, override.prompt_n, 0, 0, override.gen_n);
    } else {
        rt->prompt = load_u32_file("PMT.BIN", &rt->prompt_n);
        rt->expect = load_u32_file("EXP.BIN", &rt->expect_n);
        llm_runtime_install_request(rt, rt->prompt, rt->prompt_n, rt->expect, rt->expect_n, rt->expect_n);
    }

    llm_log_ms_metric(
        "model_load",
        llm_elapsed_us(load_start_raw, llm_now_raw_time()),
        "llm_runtime_init_assets_and_request_setup"
    );
}

static void llm_alloc_kv_caches(const struct model_cfg *cfg, float **kcache_out, float **vcache_out) {
    uint64 cache_elems = (uint64)cfg->n_layers * (uint64)cfg->runtime_seq_len * (uint64)cfg->n_kv_heads *
                         (uint64)cfg->head_dim;
    *kcache_out = llm_alloc_zeroed_floats(cache_elems);
    *vcache_out = llm_alloc_zeroed_floats(cache_elems);
}

static void __attribute__((unused)) llm_workspace_init_full(const struct model_cfg *cfg, struct llm_workspace *ws) {
    memset(ws, 0, sizeof(*ws));
    ws->hidden = llm_alloc_floats((uint64)cfg->dim);
    ws->norm_hidden = llm_alloc_floats((uint64)cfg->dim);
    ws->proj_out = llm_alloc_floats((uint64)cfg->dim);
    ws->qkv_mixed = llm_alloc_floats((uint64)cfg_full_q_rows(cfg));
    ws->query = llm_alloc_floats((uint64)cfg_attn_out_dim(cfg));
    if (cfg_has_attn_gate(cfg)) {
        ws->attn_gate = llm_alloc_floats((uint64)cfg_attn_out_dim(cfg));
    }
    ws->key = llm_alloc_floats((uint64)cfg_kv_dim(cfg));
    ws->value = llm_alloc_floats((uint64)cfg_kv_dim(cfg));
    ws->attn_out = llm_alloc_floats((uint64)cfg_attn_out_dim(cfg));
    ws->ffn_gate = llm_alloc_floats((uint64)cfg->hidden_dim);
    ws->ffn_up = llm_alloc_floats((uint64)cfg->hidden_dim);
    ws->scores = llm_alloc_floats((uint64)cfg->runtime_seq_len);
}

static void llm_apply_full_layer(
    const struct llm_runtime *rt,
    float *kcache,
    float *vcache,
    int layer_idx,
    int pos,
    struct llm_workspace *ws
) {
    const struct model_cfg *cfg = &rt->cfg;
    struct layer *ly = &rt->layers[layer_idx];
    int head_dim = cfg->head_dim;
    int attn_out_dim = cfg_attn_out_dim(cfg);
    int kv_mul = cfg_kv_mul(cfg);
    float *cosv = rt->rope + (uint64)pos * (uint64)cfg->rope_rotary_dim;
    float *sinv = cosv + cfg->rope_rotary_dim / 2;

    rmsnorm(ws->norm_hidden, ws->hidden, ly->input_norm, cfg->dim, cfg->rms_eps);
    qmatmul(ws->qkv_mixed, ly->u.full.q_proj, ws->norm_hidden);
    if (cfg_has_attn_gate(cfg)) {
        for (int h = 0; h < cfg->n_heads; h++) {
            const float *base = ws->qkv_mixed + (uint64)h * (uint64)head_dim * 2ULL;
            memcpy(ws->query + (uint64)h * (uint64)head_dim, base, (uint)head_dim * sizeof(float));
            memcpy(ws->attn_gate + (uint64)h * (uint64)head_dim, base + head_dim, (uint)head_dim * sizeof(float));
        }
    } else {
        memcpy(ws->query, ws->qkv_mixed, (uint)attn_out_dim * sizeof(float));
    }

    qmatmul(ws->key, ly->u.full.k_proj, ws->norm_hidden);
    qmatmul(ws->value, ly->u.full.v_proj, ws->norm_hidden);

    if (cfg_has_qk_norm(cfg)) {
        for (int h = 0; h < cfg->n_heads; h++) {
            rmsnorm(ws->query + (uint64)h * (uint64)head_dim,
                    ws->query + (uint64)h * (uint64)head_dim,
                    ly->u.full.q_norm, head_dim, cfg->rms_eps);
        }
        for (int h = 0; h < cfg->n_kv_heads; h++) {
            rmsnorm(ws->key + (uint64)h * (uint64)head_dim,
                    ws->key + (uint64)h * (uint64)head_dim,
                    ly->u.full.k_norm, head_dim, cfg->rms_eps);
        }
    }

    for (int h = 0; h < cfg->n_heads; h++) {
        apply_rope_partial(ws->query + (uint64)h * (uint64)head_dim, cosv, sinv, cfg->rope_rotary_dim);
    }
    for (int h = 0; h < cfg->n_kv_heads; h++) {
        apply_rope_partial(ws->key + (uint64)h * (uint64)head_dim, cosv, sinv, cfg->rope_rotary_dim);
    }

    for (int h = 0; h < cfg->n_kv_heads; h++) {
        memcpy(llm_kcache_at(kcache, cfg, layer_idx, pos, h),
               ws->key + (uint64)h * (uint64)head_dim,
               (uint)head_dim * sizeof(float));
        memcpy(llm_kcache_at(vcache, cfg, layer_idx, pos, h),
               ws->value + (uint64)h * (uint64)head_dim,
               (uint)head_dim * sizeof(float));
    }

    memset(ws->attn_out, 0, (uint)((uint64)attn_out_dim * sizeof(float)));
    float scale = 1.0f / fsqrt1((float)head_dim);
    for (int h = 0; h < cfg->n_heads; h++) {
        int kvh = h / kv_mul;
        float *qh = ws->query + (uint64)h * (uint64)head_dim;
        float *ah = ws->attn_out + (uint64)h * (uint64)head_dim;

        for (int t = 0; t <= pos; t++) {
            float *kh = llm_kcache_at(kcache, cfg, layer_idx, t, kvh);
            float acc = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                acc += qh[i] * kh[i];
            }
            ws->scores[t] = acc * scale;
        }
        softmax_inplace(ws->scores, pos + 1);

        for (int t = 0; t <= pos; t++) {
            float *vh = llm_kcache_at(vcache, cfg, layer_idx, t, kvh);
            float w = ws->scores[t];
            for (int i = 0; i < head_dim; i++) {
                ah[i] += w * vh[i];
            }
        }
    }

    if (cfg_has_attn_gate(cfg)) {
        for (int i = 0; i < attn_out_dim; i++) {
            ws->attn_out[i] *= sigmoid_approx(ws->attn_gate[i]);
        }
    }

    qmatmul(ws->proj_out, ly->u.full.o_proj, ws->attn_out);
    for (int i = 0; i < cfg->dim; i++) {
        ws->hidden[i] += ws->proj_out[i];
    }
}

static void llm_apply_ffn(const struct model_cfg *cfg, struct layer *ly, struct llm_workspace *ws) {
    rmsnorm(ws->norm_hidden, ws->hidden, ly->post_norm, cfg->dim, cfg->rms_eps);
    qmatmul(ws->ffn_gate, ly->gate_proj, ws->norm_hidden);
    qmatmul(ws->ffn_up, ly->up_proj, ws->norm_hidden);
    for (int i = 0; i < cfg->hidden_dim; i++) {
        ws->ffn_gate[i] = silu_approx(ws->ffn_gate[i]) * ws->ffn_up[i];
    }
    qmatmul(ws->proj_out, ly->down_proj, ws->ffn_gate);
    for (int i = 0; i < cfg->dim; i++) {
        ws->hidden[i] += ws->proj_out[i];
    }
}

static void llm_drive_decode(
    struct llm_runtime *rt,
    struct llm_workspace *ws,
    float *kcache,
    float *vcache,
    float *linear_conv_cache,
    float *linear_state_cache,
    llm_token_forward_fn token_forward
) {
    int total_steps = rt->prompt_n + rt->gen_n - 1;
    int mismatch = 0;
    uint64 decode_start_raw = llm_now_raw_time();
    uint64 first_token_raw = 0;

    LLM_LOG("starting forward pass\n");
    for (int pos = 0; pos < total_steps; pos++) {
        int next = token_forward(rt, kcache, vcache, linear_conv_cache, linear_state_cache, pos, ws);
        if (pos >= rt->prompt_n - 1) {
            int gen_idx = pos - (rt->prompt_n - 1);
            rt->seq[rt->prompt_n + gen_idx] = (uint32)next;
            if (gen_idx == 0) {
                first_token_raw = llm_now_raw_time();
            }
            int has_expect = rt->expect != 0 && gen_idx < rt->expect_n;
            int token_mismatch = has_expect && (uint32)next != rt->expect[gen_idx];
            if (token_mismatch) {
                mismatch = 1;
            }
            if (llm_log_enabled()) {
                LLM_LOG("gen[%d] token=%d", gen_idx, next);
                if (has_expect) {
                    printf(" expect=%d", rt->expect[gen_idx]);
                    if (token_mismatch) {
                        printf(" note=host-mismatch");
                    }
                }
                printf("\n");
            }
        }
    }

    write_service_output(rt->seq, rt->prompt_n, rt->gen_n);
    uint64 decode_end_raw = llm_now_raw_time();
    uint64 decode_total_us = llm_elapsed_us(decode_start_raw, decode_end_raw);
    uint64 ttft_us = first_token_raw ? llm_elapsed_us(decode_start_raw, first_token_raw) : 0;
    uint64 steady_us = (rt->gen_n > 1 && first_token_raw) ? llm_elapsed_us(first_token_raw, decode_end_raw) : 0;
    uint64 tpot_us = (rt->gen_n > 1 && steady_us > 0) ? llm_avg_us(steady_us, rt->gen_n - 1) : 0;
    uint64 overall_tps_x100 = llm_tokens_per_s_x100(rt->gen_n, decode_total_us);
    uint64 steady_tps_x100 = (rt->gen_n > 1) ? llm_tokens_per_s_x100(rt->gen_n - 1, steady_us) : 0;

    LLM_LOG("metrics prompt_tokens=%d gen_tokens=%d total_steps=%d\n", rt->prompt_n, rt->gen_n, total_steps);
    llm_log_ms_metric("decode_total", decode_total_us, "decode_start_to_last_generated_token");
    if (ttft_us > 0) {
        llm_log_ms_metric("ttft", ttft_us, "decode_start_to_first_generated_token_includes_prefill");
    } else {
        LLM_LOG("ttft_ms=n/a scope=decode_start_to_first_generated_token_includes_prefill\n");
    }
    if (rt->gen_n > 1 && steady_us > 0) {
        llm_log_ms_metric("tpot", tpot_us, "average_time_for_tokens_2_to_N_excludes_ttft");
        llm_log_rate_metric("steady_token_throughput", steady_tps_x100, "tokens_2_to_N_over_post_first_token_window");
    } else {
        LLM_LOG("tpot_ms=n/a scope=average_time_for_tokens_2_to_N_excludes_ttft\n");
        LLM_LOG("steady_token_throughput_tok_per_s=n/a scope=tokens_2_to_N_over_post_first_token_window\n");
    }
    llm_log_rate_metric("token_throughput", overall_tps_x100, "all_generated_tokens_over_full_decode_window_including_ttft");
    if (mismatch) {
        LLM_LOG("completed with host-reference mismatch\n");
    } else {
        LLM_LOG("completed\n");
    }
}

#endif
