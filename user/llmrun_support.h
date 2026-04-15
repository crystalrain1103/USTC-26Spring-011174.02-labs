#ifndef __LLMRUN_SUPPORT_H__
#define __LLMRUN_SUPPORT_H__

#include "user.h"
#include "stat.h"

#define LLM_CFG_MAGIC 0x31474643U
#define LLM_CFG_VERSION_V1 1U
#define LLM_CFG_VERSION_V2 2U

#define MODEL_KIND_SMOL 0U
#define MODEL_KIND_QWEN 1U

#define LAYER_KIND_FULL 0U
#define LAYER_KIND_LINEAR 1U

#define CFG_FLAG_ATTN_GATE 0x1U
#define CFG_FLAG_QK_NORM 0x2U

#define LEGACY_DIM 576
#define LEGACY_HIDDEN_DIM 1536
#define LEGACY_N_LAYERS 30
#define LEGACY_N_HEADS 9
#define LEGACY_N_KV_HEADS 3
#define LEGACY_HEAD_DIM 64
#define LEGACY_VOCAB_SIZE 49152
#define LEGACY_SEQ_LEN 2048
#define LEGACY_RUNTIME_SEQ_LEN 256
#define LEGACY_RMS_EPS 1e-5f
#define LEGACY_ROPE_THETA 100000.0f

#define LINEAR_NORM_EPS 1e-6f

struct model_cfg {
    uint32 model_kind;
    uint32 flags;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int seq_len;
    int runtime_seq_len;
    float rms_eps;
    float rope_theta;
    int rope_rotary_dim;
    int linear_num_k_heads;
    int linear_num_v_heads;
    int linear_key_head_dim;
    int linear_value_head_dim;
    int linear_conv_kernel;
};

struct model_cfg_disk_v1 {
    uint32 magic;
    uint32 version;
    uint32 dim;
    uint32 hidden_dim;
    uint32 n_layers;
    uint32 n_heads;
    uint32 n_kv_heads;
    uint32 vocab_size;
    uint32 seq_len;
    uint32 runtime_seq_len;
    float rms_eps;
    float rope_theta;
} __attribute__((packed));

struct model_cfg_disk_v2 {
    uint32 magic;
    uint32 version;
    uint32 model_kind;
    uint32 flags;
    uint32 dim;
    uint32 hidden_dim;
    uint32 n_layers;
    uint32 n_heads;
    uint32 n_kv_heads;
    uint32 head_dim;
    uint32 vocab_size;
    uint32 seq_len;
    uint32 runtime_seq_len;
    float rms_eps;
    float rope_theta;
    uint32 rope_rotary_dim;
    uint32 linear_num_k_heads;
    uint32 linear_num_v_heads;
    uint32 linear_key_head_dim;
    uint32 linear_value_head_dim;
    uint32 linear_conv_kernel;
} __attribute__((packed));

struct qmat {
    int rows;
    int cols;
    float *scales;
    int8 *q;
};

struct full_layer {
    float *q_norm;
    float *k_norm;
    struct qmat q_proj;
    struct qmat k_proj;
    struct qmat v_proj;
    struct qmat o_proj;
};

struct linear_layer {
    float *conv_weight;
    float *dt_bias;
    float *a_log;
    float *norm;
    struct qmat qkv_proj;
    struct qmat z_proj;
    struct qmat a_proj;
    struct qmat b_proj;
    struct qmat out_proj;
};

struct layer {
    uint32 kind;
    float *input_norm;
    float *post_norm;
    struct qmat gate_proj;
    struct qmat up_proj;
    struct qmat down_proj;
    union {
        struct full_layer full;
        struct linear_layer linear;
    } u;
};

/*
 * The workspace groups all temporary vectors used during one token forward.
 * Students can treat it as "scratch memory" for the model runtime.
 */
struct llm_workspace {
    float *hidden;
    float *norm_hidden;
    float *proj_out;
    float *qkv_mixed;
    float *query;
    float *attn_gate;
    float *key;
    float *value;
    float *linear_gate;
    float *linear_a;
    float *linear_b;
    float *attn_out;
    float *ffn_gate;
    float *ffn_up;
    float *scores;
};

const char *llm_prog_name(void);
void llm_set_prog_name(const char *name);
void llm_set_prog_name_from_argv(const char *argv0, const char *fallback);

#define LLM_LOG(...)                     \
    do {                                \
        printf("[%s] ", llm_prog_name()); \
        printf(__VA_ARGS__);            \
    } while (0)

#define LLM_ERR(...)                    \
    do {                                \
        fprintf(2, "%s: ", llm_prog_name()); \
        fprintf(2, __VA_ARGS__);        \
    } while (0)

int max_int(int a, int b);

/* Model-shape helpers. */
int cfg_has_attn_gate(const struct model_cfg *cfg);
int cfg_has_qk_norm(const struct model_cfg *cfg);
int cfg_attn_out_dim(const struct model_cfg *cfg);
int cfg_full_q_rows(const struct model_cfg *cfg);
int cfg_kv_dim(const struct model_cfg *cfg);
int cfg_kv_mul(const struct model_cfg *cfg);
int cfg_linear_key_dim(const struct model_cfg *cfg);
int cfg_linear_value_dim(const struct model_cfg *cfg);
int cfg_linear_conv_dim(const struct model_cfg *cfg);

/* Memory / file helpers. */
void *xmalloc(uint64 n);
int read_full(int fd, void *buf, int n);
void write_service_output(uint32 *seq, int prompt_n, int gen_n);

char *load_file(const char *path, uint64 *size_out);
void load_model_cfg(struct model_cfg *cfg);
uint32 *load_u32_file(const char *path, int *count_out);
uint32 *load_layer_kinds(const struct model_cfg *cfg);

/* Math helpers used by the inference path. */
float fsqrt1(float x);
float exp_approx(float x);
float softplus_approx(float x);
float sigmoid_approx(float x);
float silu_approx(float x);
void softmax_inplace(float *x, int n);
void rmsnorm(float *out, const float *x, const float *weight, int n, float eps);
void l2norm_inplace(float *x, int n, float eps);
void qmatmul(float *out, struct qmat m, const float *x);
void apply_rope_partial(float *vec, const float *cosv, const float *sinv, int rotary_dim);

/* Layer/blob parsing helpers. */
void parse_layer_blob(struct layer *ly, char *blob, uint64 size, const struct model_cfg *cfg, uint32 kind);
void make_layer_path(int layer, char *out, int out_sz);
int argmax_embed(struct qmat emb, const float *x);

#endif
