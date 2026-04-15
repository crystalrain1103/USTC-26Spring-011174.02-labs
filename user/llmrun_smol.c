#include "llmrun_runtime.h"

static int token_forward(
    const struct llm_runtime *rt,
    float *kcache,
    float *vcache,
    float *linear_conv_cache,
    float *linear_state_cache,
    int pos,
    struct llm_workspace *ws
) {
    (void)linear_conv_cache;
    (void)linear_state_cache;

    const struct model_cfg *cfg = &rt->cfg;
    uint32 token = rt->seq[pos];
    const int8 *erow = rt->emb.q + (uint64)token * (uint64)rt->emb.cols;

    for (int i = 0; i < cfg->dim; i++) {
        ws->hidden[i] = (float)erow[i] * rt->emb.scales[token];
    }

    for (int l = 0; l < cfg->n_layers; l++) {
        llm_apply_full_layer(rt, kcache, vcache, l, pos, ws);
        llm_apply_ffn(cfg, &rt->layers[l], ws);
    }

    rmsnorm(ws->norm_hidden, ws->hidden, rt->final_norm, cfg->dim, cfg->rms_eps);
    return argmax_embed(rt->emb, ws->norm_hidden);
}

int main(int argc, char *argv[]) {
    llm_set_prog_name_from_argv(argc > 0 ? argv[0] : 0, "llmrun_smol");
    const char *asset_dir = argc > 1 ? argv[1] : "/AI/SMOL";
    if (chdir(asset_dir) < 0) {
        LLM_ERR("failed to chdir to %s\n", asset_dir);
        exit(1);
    }

    struct llm_runtime rt;
    struct llm_workspace ws;
    float *kcache = 0;
    float *vcache = 0;

    llm_runtime_init(&rt, MODEL_KIND_SMOL, 0);
    llm_alloc_kv_caches(&rt.cfg, &kcache, &vcache);
    llm_workspace_init_full(&rt.cfg, &ws);
    llm_drive_decode(&rt, &ws, kcache, vcache, 0, 0, token_forward);
    exit(0);
}
