#include "llmrun_runtime.h"

static float *linear_conv_state_at(float *cache, const struct model_cfg *cfg, int layer, int channel) {
    uint64 idx = ((uint64)layer * (uint64)cfg_linear_conv_dim(cfg) + (uint64)channel) * (uint64)cfg->linear_conv_kernel;
    return cache + idx;
}

static float *linear_state_at(float *cache, const struct model_cfg *cfg, int layer, int head) {
    uint64 idx = ((uint64)layer * (uint64)cfg->linear_num_v_heads + (uint64)head) *
                 (uint64)cfg->linear_key_head_dim * (uint64)cfg->linear_value_head_dim;
    return cache + idx;
}

static void apply_linear_layer(
    const struct llm_runtime *rt,
    float *linear_conv_cache,
    float *linear_state_cache,
    int layer_idx,
    struct llm_workspace *ws
) {
    const struct model_cfg *cfg = &rt->cfg;
    struct layer *ly = &rt->layers[layer_idx];
    int conv_dim = cfg_linear_conv_dim(cfg);
    int value_dim = cfg_linear_value_dim(cfg);
    int v_heads = cfg->linear_num_v_heads;
    int k_heads = cfg->linear_num_k_heads;
    int key_head_dim = cfg->linear_key_head_dim;
    int value_head_dim = cfg->linear_value_head_dim;
    int kernel = cfg->linear_conv_kernel;
    int head_rep = v_heads / k_heads;
    float *delta = ws->attn_gate;

    rmsnorm(ws->norm_hidden, ws->hidden, ly->input_norm, cfg->dim, cfg->rms_eps);
    qmatmul(ws->qkv_mixed, ly->u.linear.qkv_proj, ws->norm_hidden);
    for (int c = 0; c < conv_dim; c++) {
        float *state = linear_conv_state_at(linear_conv_cache, cfg, layer_idx, c);
        float acc = 0.0f;
        float *w = ly->u.linear.conv_weight + (uint64)c * (uint64)kernel;

        for (int j = 0; j < kernel - 1; j++) {
            state[j] = state[j + 1];
        }
        state[kernel - 1] = ws->qkv_mixed[c];
        for (int j = 0; j < kernel; j++) {
            acc += w[j] * state[j];
        }
        ws->qkv_mixed[c] = silu_approx(acc);
    }

    memcpy(ws->query, ws->qkv_mixed, (uint)cfg_linear_key_dim(cfg) * sizeof(float));
    memcpy(ws->key, ws->qkv_mixed + cfg_linear_key_dim(cfg), (uint)cfg_linear_key_dim(cfg) * sizeof(float));
    memcpy(ws->value, ws->qkv_mixed + cfg_linear_key_dim(cfg) * 2, (uint)value_dim * sizeof(float));

    qmatmul(ws->linear_gate, ly->u.linear.z_proj, ws->norm_hidden);
    qmatmul(ws->linear_a, ly->u.linear.a_proj, ws->norm_hidden);
    qmatmul(ws->linear_b, ly->u.linear.b_proj, ws->norm_hidden);
    memset(ws->attn_out, 0, (uint)((uint64)value_dim * sizeof(float)));

    float scale = 1.0f / fsqrt1((float)key_head_dim);
    for (int h = 0; h < v_heads; h++) {
        int src_head = h / head_rep;
        float *qh = ws->query + (uint64)src_head * (uint64)key_head_dim;
        float *kh = ws->key + (uint64)src_head * (uint64)key_head_dim;
        float *vh = ws->value + (uint64)h * (uint64)value_head_dim;
        float *zh = ws->linear_gate + (uint64)h * (uint64)value_head_dim;
        float *out_head = ws->attn_out + (uint64)h * (uint64)value_head_dim;
        float *state = linear_state_at(linear_state_cache, cfg, layer_idx, h);

        memcpy(ws->ffn_gate, qh, (uint)key_head_dim * sizeof(float));
        memcpy(ws->ffn_up, kh, (uint)key_head_dim * sizeof(float));
        l2norm_inplace(ws->ffn_gate, key_head_dim, LINEAR_NORM_EPS);
        l2norm_inplace(ws->ffn_up, key_head_dim, LINEAR_NORM_EPS);
        for (int i = 0; i < key_head_dim; i++) {
            ws->ffn_gate[i] *= scale;
        }

        float beta = sigmoid_approx(ws->linear_b[h]);
        float g = -exp_approx(ly->u.linear.a_log[h]) * softplus_approx(ws->linear_a[h] + ly->u.linear.dt_bias[h]);
        float gexp = exp_approx(g);
        for (int i = 0; i < key_head_dim * value_head_dim; i++) {
            state[i] *= gexp;
        }

        for (int v = 0; v < value_head_dim; v++) {
            float mem = 0.0f;
            for (int i = 0; i < key_head_dim; i++) {
                mem += state[(uint64)i * (uint64)value_head_dim + (uint64)v] * ws->ffn_up[i];
            }
            delta[v] = (vh[v] - mem) * beta;
        }

        for (int i = 0; i < key_head_dim; i++) {
            float *row = state + (uint64)i * (uint64)value_head_dim;
            float kval = ws->ffn_up[i];
            float qval = ws->ffn_gate[i];
            for (int v = 0; v < value_head_dim; v++) {
                row[v] += kval * delta[v];
                out_head[v] += row[v] * qval;
            }
        }

        rmsnorm(out_head, out_head, ly->u.linear.norm, value_head_dim, cfg->rms_eps);
        for (int v = 0; v < value_head_dim; v++) {
            out_head[v] *= silu_approx(zh[v]);
        }
    }

    qmatmul(ws->proj_out, ly->u.linear.out_proj, ws->attn_out);
    for (int i = 0; i < cfg->dim; i++) {
        ws->hidden[i] += ws->proj_out[i];
    }
}

static void qwen_alloc_linear_caches(const struct model_cfg *cfg, float **linear_conv_cache, float **linear_state_cache) {
    uint64 conv_elems = (uint64)cfg->n_layers * (uint64)cfg_linear_conv_dim(cfg) * (uint64)cfg->linear_conv_kernel;
    uint64 state_elems = (uint64)cfg->n_layers * (uint64)cfg->linear_num_v_heads *
                         (uint64)cfg->linear_key_head_dim * (uint64)cfg->linear_value_head_dim;

    *linear_conv_cache = llm_alloc_zeroed_floats(conv_elems);
    *linear_state_cache = llm_alloc_zeroed_floats(state_elems);
}

static void qwen_workspace_init(const struct model_cfg *cfg, struct llm_workspace *ws) {
    int mixed_qkv_cap = max_int(cfg_full_q_rows(cfg), cfg_linear_conv_dim(cfg));
    int query_cap = max_int(cfg_attn_out_dim(cfg), cfg_linear_key_dim(cfg));
    int gate_cap = max_int(cfg_attn_out_dim(cfg), cfg_linear_value_dim(cfg));
    int key_cap = max_int(cfg_kv_dim(cfg), cfg_linear_key_dim(cfg));
    int value_cap = max_int(cfg_kv_dim(cfg), cfg_linear_value_dim(cfg));
    int attention_out_cap = max_int(cfg_attn_out_dim(cfg), cfg_linear_value_dim(cfg));

    memset(ws, 0, sizeof(*ws));
    ws->hidden = llm_alloc_floats((uint64)cfg->dim);
    ws->norm_hidden = llm_alloc_floats((uint64)cfg->dim);
    ws->proj_out = llm_alloc_floats((uint64)cfg->dim);
    ws->qkv_mixed = llm_alloc_floats((uint64)mixed_qkv_cap);
    ws->query = llm_alloc_floats((uint64)query_cap);
    ws->attn_gate = llm_alloc_floats((uint64)gate_cap);
    ws->key = llm_alloc_floats((uint64)key_cap);
    ws->value = llm_alloc_floats((uint64)value_cap);
    ws->linear_gate = llm_alloc_floats((uint64)cfg_linear_value_dim(cfg));
    ws->linear_a = llm_alloc_floats((uint64)cfg->linear_num_v_heads);
    ws->linear_b = llm_alloc_floats((uint64)cfg->linear_num_v_heads);
    ws->attn_out = llm_alloc_floats((uint64)attention_out_cap);
    ws->ffn_gate = llm_alloc_floats((uint64)cfg->hidden_dim);
    ws->ffn_up = llm_alloc_floats((uint64)cfg->hidden_dim);
    ws->scores = llm_alloc_floats((uint64)cfg->runtime_seq_len);
}

static int token_forward(
    const struct llm_runtime *rt,
    float *kcache,
    float *vcache,
    float *linear_conv_cache,
    float *linear_state_cache,
    int pos,
    struct llm_workspace *ws
) {
    const struct model_cfg *cfg = &rt->cfg;
    uint32 token = rt->seq[pos];
    const int8 *erow = rt->emb.q + (uint64)token * (uint64)rt->emb.cols;

    for (int i = 0; i < cfg->dim; i++) {
        ws->hidden[i] = (float)erow[i] * rt->emb.scales[token];
    }

    for (int l = 0; l < cfg->n_layers; l++) {
        if (rt->layer_kinds[l] == LAYER_KIND_FULL) {
            llm_apply_full_layer(rt, kcache, vcache, l, pos, ws);
        } else if (rt->layer_kinds[l] == LAYER_KIND_LINEAR) {
            apply_linear_layer(rt, linear_conv_cache, linear_state_cache, l, ws);
        } else {
            LLM_ERR("unsupported layer kind %d\n", (int)rt->layer_kinds[l]);
            exit(1);
        }
        llm_apply_ffn(cfg, &rt->layers[l], ws);
    }

    rmsnorm(ws->norm_hidden, ws->hidden, rt->final_norm, cfg->dim, cfg->rms_eps);
    return argmax_embed(rt->emb, ws->norm_hidden);
}

int main(int argc, char *argv[]) {
    (void)argc;

    llm_set_prog_name_from_argv(argc > 0 ? argv[0] : 0, "llmrun_qwen");

    struct llm_runtime rt;
    struct llm_workspace ws;
    float *kcache = 0;
    float *vcache = 0;
    float *linear_conv_cache = 0;
    float *linear_state_cache = 0;

    llm_runtime_init(&rt, MODEL_KIND_QWEN, 1, argc, argv);
    llm_alloc_kv_caches(&rt.cfg, &kcache, &vcache);
    qwen_alloc_linear_caches(&rt.cfg, &linear_conv_cache, &linear_state_cache);
    qwen_workspace_init(&rt.cfg, &ws);
    llm_drive_decode(&rt, &ws, kcache, vcache, linear_conv_cache, linear_state_cache, token_forward);
    exit(0);
}
