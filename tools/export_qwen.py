#!/usr/bin/env python3
"""
Export Qwen3.5 text-only weights into the teaching-kernel blob format.

This path is intentionally separated from the required SmolLM bring-up path.
It exports only the language-model subgraph and ignores visual/MTP weights.

The goal is not to reproduce the full multimodal checkpoint. The goal is to
give the course a heavier bonus model whose runtime pressure makes later
memory-management and storage-design decisions feel necessary rather than
theoretical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from export_smollm import (
    exp_approx,
    pack_qmat,
    qmatmul,
    rmsnorm,
    rowwise_q8,
    sigmoid_approx,
    silu_approx,
    softmax_approx,
    write_text_sidecars,
)

CFG_MAGIC = 0x31474643
CFG_VERSION = 2

MODEL_KIND_QWEN = 1

LAYER_FULL = 0
LAYER_LINEAR = 1

CFG_FLAG_ATTN_GATE = 0x1
CFG_FLAG_QK_NORM = 0x2

DEFAULT_RUNTIME_SEQ = 64


@dataclass(frozen=True)
class QwenConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    seq_len: int
    rms_eps: float
    rope_theta: float
    rope_rotary_dim: int
    linear_num_k_heads: int
    linear_num_v_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    layer_types: tuple[str, ...]

    @property
    def attn_out_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_k_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_v_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_key_dim * 2 + self.linear_value_dim


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_config(model_dir: Path) -> QwenConfig:
    raw = read_json(model_dir / "config.json")
    if raw.get("model_type") != "qwen3_5":
        raise ValueError("only qwen3_5 exports are supported")
    text = raw["text_config"]
    rope = text["rope_parameters"]
    layer_types = tuple(text["layer_types"])
    return QwenConfig(
        dim=int(text["hidden_size"]),
        hidden_dim=int(text["intermediate_size"]),
        n_layers=int(text["num_hidden_layers"]),
        n_heads=int(text["num_attention_heads"]),
        n_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        vocab_size=int(text["vocab_size"]),
        seq_len=int(text["max_position_embeddings"]),
        rms_eps=float(text["rms_norm_eps"]),
        rope_theta=float(rope["rope_theta"]),
        rope_rotary_dim=int(int(text["head_dim"]) * float(rope.get("partial_rotary_factor", 1.0))),
        linear_num_k_heads=int(text["linear_num_key_heads"]),
        linear_num_v_heads=int(text["linear_num_value_heads"]),
        linear_key_head_dim=int(text["linear_key_head_dim"]),
        linear_value_head_dim=int(text["linear_value_head_dim"]),
        linear_conv_kernel=int(text["linear_conv_kernel_dim"]),
        layer_types=layer_types,
    )


def tensor_to_f32(sf: safe_open, name: str) -> np.ndarray:
    return sf.get_tensor(name).to(dtype=torch.float32).cpu().numpy()


def load_named_tensors(model_dir: Path, wanted: list[str]) -> dict[str, np.ndarray]:
    index = read_json(model_dir / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name in wanted:
        file_name = weight_map.get(name)
        if file_name is None:
            raise KeyError(f"missing tensor {name}")
        by_file.setdefault(file_name, []).append(name)

    out: dict[str, np.ndarray] = {}
    for file_name, names in by_file.items():
        with safe_open(str(model_dir / file_name), framework="pt", device="cpu") as sf:
            for name in names:
                out[name] = tensor_to_f32(sf, name)
    return out


def qwen_rms_weight(arr: np.ndarray) -> np.ndarray:
    return (np.asarray(arr, dtype=np.float32) + 1.0).astype(np.float32, copy=False)


def layer_code(layer_type: str) -> int:
    if layer_type == "full_attention":
        return LAYER_FULL
    if layer_type == "linear_attention":
        return LAYER_LINEAR
    raise ValueError(f"unsupported layer type {layer_type}")


def write_layer_types(path: Path, cfg: QwenConfig) -> None:
    raw = struct.pack("<I", cfg.n_layers)
    raw += struct.pack("<" + "I" * cfg.n_layers, *(layer_code(t) for t in cfg.layer_types))
    path.write_bytes(raw)


def write_u32_file(path: Path, values: list[int]) -> None:
    raw = struct.pack("<I", len(values))
    if values:
        raw += struct.pack("<" + "I" * len(values), *values)
    path.write_bytes(raw)


def write_cfg_file(path: Path, cfg: QwenConfig, runtime_seq_len: int) -> None:
    raw = struct.pack(
        "<13I2f6I",
        CFG_MAGIC,
        CFG_VERSION,
        MODEL_KIND_QWEN,
        CFG_FLAG_ATTN_GATE | CFG_FLAG_QK_NORM,
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.seq_len,
        runtime_seq_len,
        cfg.rms_eps,
        cfg.rope_theta,
        cfg.rope_rotary_dim,
        cfg.linear_num_k_heads,
        cfg.linear_num_v_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        cfg.linear_conv_kernel,
    )
    path.write_bytes(raw)


def rope_angles_qwen(pos: int, rotary_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    idx = np.arange(half, dtype=np.float32)
    freqs = pos / np.power(theta, (2.0 * idx) / rotary_dim)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def write_rope_file(path: Path, cfg: QwenConfig, runtime_seq_len: int) -> None:
    raw = bytearray()
    for pos in range(runtime_seq_len):
        cosv, sinv = rope_angles_qwen(pos, cfg.rope_rotary_dim, cfg.rope_theta)
        raw.extend(np.asarray(cosv, dtype="<f4").tobytes())
        raw.extend(np.asarray(sinv, dtype="<f4").tobytes())
    path.write_bytes(bytes(raw))


def apply_rope_partial_inplace(vec: np.ndarray, cos: np.ndarray, sin: np.ndarray, rotary_dim: int) -> None:
    half = rotary_dim // 2
    x1 = vec[:half].copy()
    x2 = vec[half:rotary_dim].copy()
    vec[:half] = x1 * cos - x2 * sin
    vec[half:rotary_dim] = x1 * sin + x2 * cos


def softplus_approx(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pos = x >= 0.0
    out = np.empty_like(x)
    ep = exp_approx(-x[pos])
    en = exp_approx(x[~pos])
    out[pos] = x[pos] + (ep - 0.5 * ep * ep + (ep * ep * ep) / 3.0 - 0.25 * ep**4 + 0.2 * ep**5)
    out[~pos] = en - 0.5 * en * en + (en * en * en) / 3.0 - 0.25 * en**4 + 0.2 * en**5
    return out.astype(np.float32, copy=False)


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    inv = np.float32(1.0 / np.sqrt(np.sum(x * x) + eps))
    return x * inv


def split_qwen_query_and_gate(raw: np.ndarray, cfg: QwenConfig) -> tuple[np.ndarray, np.ndarray]:
    out_q = np.empty((cfg.n_heads, cfg.head_dim), dtype=np.float32)
    out_g = np.empty((cfg.n_heads, cfg.head_dim), dtype=np.float32)
    stride = cfg.head_dim * 2
    for h in range(cfg.n_heads):
        base = h * stride
        out_q[h] = raw[base : base + cfg.head_dim]
        out_g[h] = raw[base + cfg.head_dim : base + stride]
    return out_q, out_g


def chunked_argmax_embed(scales: np.ndarray, q: np.ndarray, x: np.ndarray, chunk_rows: int = 1024) -> int:
    best_idx = 0
    best_val = -1.0e30
    for start in range(0, q.shape[0], chunk_rows):
        end = min(start + chunk_rows, q.shape[0])
        chunk = q[start:end].astype(np.float32)
        scores = chunk @ x
        scores *= scales[start:end]
        idx = int(np.argmax(scores))
        val = float(scores[idx])
        if val > best_val:
            best_val = val
            best_idx = start + idx
    return best_idx


def blob_full_layer(weights: dict[str, np.ndarray], qweights: dict[str, tuple[np.ndarray, np.ndarray]], layer: int) -> bytes:
    p = f"model.language_model.layers.{layer}."
    parts: list[bytes] = []
    parts.append(np.asarray(qwen_rms_weight(weights[p + "input_layernorm.weight"]), dtype="<f4").tobytes())
    parts.append(np.asarray(qwen_rms_weight(weights[p + "post_attention_layernorm.weight"]), dtype="<f4").tobytes())
    parts.append(np.asarray(qwen_rms_weight(weights[p + "self_attn.q_norm.weight"]), dtype="<f4").tobytes())
    parts.append(np.asarray(qwen_rms_weight(weights[p + "self_attn.k_norm.weight"]), dtype="<f4").tobytes())
    for name in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]:
        scales, q = qweights[p + name]
        parts.append(pack_qmat(scales, q))
    return b"".join(parts)


def blob_linear_layer(weights: dict[str, np.ndarray], qweights: dict[str, tuple[np.ndarray, np.ndarray]], layer: int) -> bytes:
    p = f"model.language_model.layers.{layer}."
    parts: list[bytes] = []
    parts.append(np.asarray(qwen_rms_weight(weights[p + "input_layernorm.weight"]), dtype="<f4").tobytes())
    parts.append(np.asarray(qwen_rms_weight(weights[p + "post_attention_layernorm.weight"]), dtype="<f4").tobytes())
    conv = np.asarray(weights[p + "linear_attn.conv1d.weight"].reshape(-1, weights[p + "linear_attn.conv1d.weight"].shape[-1]), dtype="<f4")
    parts.append(conv.tobytes())
    parts.append(np.asarray(weights[p + "linear_attn.dt_bias"], dtype="<f4").tobytes())
    parts.append(np.asarray(weights[p + "linear_attn.A_log"], dtype="<f4").tobytes())
    parts.append(np.asarray(weights[p + "linear_attn.norm.weight"], dtype="<f4").tobytes())
    for name in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.out_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]:
        scales, q = qweights[p + name]
        parts.append(pack_qmat(scales, q))
    return b"".join(parts)


def build_runtime_weights(
    model_dir: Path, cfg: QwenConfig
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    wanted: list[str] = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ]
    for i, layer_type in enumerate(cfg.layer_types):
        p = f"model.language_model.layers.{i}."
        wanted.extend(
            [
                p + "input_layernorm.weight",
                p + "post_attention_layernorm.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
        )
        if layer_type == "full_attention":
            wanted.extend(
                [
                    p + "self_attn.q_proj.weight",
                    p + "self_attn.k_proj.weight",
                    p + "self_attn.v_proj.weight",
                    p + "self_attn.o_proj.weight",
                    p + "self_attn.q_norm.weight",
                    p + "self_attn.k_norm.weight",
                ]
            )
        elif layer_type == "linear_attention":
            wanted.extend(
                [
                    p + "linear_attn.conv1d.weight",
                    p + "linear_attn.dt_bias",
                    p + "linear_attn.A_log",
                    p + "linear_attn.norm.weight",
                    p + "linear_attn.in_proj_qkv.weight",
                    p + "linear_attn.in_proj_z.weight",
                    p + "linear_attn.in_proj_a.weight",
                    p + "linear_attn.in_proj_b.weight",
                    p + "linear_attn.out_proj.weight",
                ]
            )
        else:
            raise ValueError(f"unsupported layer type {layer_type}")

    weights = load_named_tensors(model_dir, wanted)
    qweights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, arr in weights.items():
        if arr.ndim == 2:
            qweights[name] = rowwise_q8(arr)

    emb_scales, emb_q = qweights["model.language_model.embed_tokens.weight"]
    final_norm = qwen_rms_weight(weights["model.language_model.norm.weight"])
    return weights, qweights, emb_scales, emb_q, final_norm


def forward_quantized(
    cfg: QwenConfig,
    weights: dict[str, np.ndarray],
    qweights: dict[str, tuple[np.ndarray, np.ndarray]],
    emb_scales: np.ndarray,
    emb_q: np.ndarray,
    final_norm: np.ndarray,
    tokens: list[int],
    predict: int,
) -> list[int]:
    cache_k = np.zeros((cfg.n_layers, len(tokens) + predict, cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
    cache_v = np.zeros((cfg.n_layers, len(tokens) + predict, cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
    conv_state = np.zeros((cfg.n_layers, cfg.linear_conv_dim, cfg.linear_conv_kernel), dtype=np.float32)
    recurrent_state = np.zeros(
        (cfg.n_layers, cfg.linear_num_v_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
        dtype=np.float32,
    )
    out: list[int] = []

    for pos in range(len(tokens) + predict - 1):
        token = tokens[pos] if pos < len(tokens) else out[pos - len(tokens)]
        x = emb_q[token].astype(np.float32) * emb_scales[token]

        cos, sin = rope_angles_qwen(pos, cfg.rope_rotary_dim, cfg.rope_theta)
        for layer, layer_type in enumerate(cfg.layer_types):
            p = f"model.language_model.layers.{layer}."
            xb = rmsnorm(x, qwen_rms_weight(weights[p + "input_layernorm.weight"]), cfg.rms_eps)

            if layer_type == "full_attention":
                q_sc, q_q = qweights[p + "self_attn.q_proj.weight"]
                k_sc, k_q = qweights[p + "self_attn.k_proj.weight"]
                v_sc, v_q = qweights[p + "self_attn.v_proj.weight"]
                o_sc, o_q = qweights[p + "self_attn.o_proj.weight"]

                qraw = qmatmul(q_sc, q_q, xb)
                qv, gate = split_qwen_query_and_gate(qraw, cfg)
                kv = qmatmul(k_sc, k_q, xb).reshape(cfg.n_kv_heads, cfg.head_dim)
                vv = qmatmul(v_sc, v_q, xb).reshape(cfg.n_kv_heads, cfg.head_dim)

                q_norm = qwen_rms_weight(weights[p + "self_attn.q_norm.weight"])
                k_norm = qwen_rms_weight(weights[p + "self_attn.k_norm.weight"])
                for h in range(cfg.n_heads):
                    qv[h] = rmsnorm(qv[h], q_norm, cfg.rms_eps)
                    apply_rope_partial_inplace(qv[h], cos, sin, cfg.rope_rotary_dim)
                for h in range(cfg.n_kv_heads):
                    kv[h] = rmsnorm(kv[h], k_norm, cfg.rms_eps)
                    apply_rope_partial_inplace(kv[h], cos, sin, cfg.rope_rotary_dim)

                cache_k[layer, pos] = kv
                cache_v[layer, pos] = vv

                att = np.zeros((cfg.n_heads, cfg.head_dim), dtype=np.float32)
                scale = np.float32(1.0 / np.sqrt(cfg.head_dim))
                for h in range(cfg.n_heads):
                    kv_h = h // cfg.kv_mul
                    scores = np.zeros(pos + 1, dtype=np.float32)
                    for t in range(pos + 1):
                        scores[t] = np.dot(qv[h], cache_k[layer, t, kv_h]) * scale
                    probs = softmax_approx(scores)
                    for t in range(pos + 1):
                        att[h] += probs[t] * cache_v[layer, t, kv_h]

                att_flat = att.reshape(cfg.attn_out_dim)
                att_flat *= sigmoid_approx(gate.reshape(cfg.attn_out_dim))
                x = x + qmatmul(o_sc, o_q, att_flat)
            elif layer_type == "linear_attention":
                qkv_sc, qkv_q = qweights[p + "linear_attn.in_proj_qkv.weight"]
                z_sc, z_q = qweights[p + "linear_attn.in_proj_z.weight"]
                a_sc, a_q = qweights[p + "linear_attn.in_proj_a.weight"]
                b_sc, b_q = qweights[p + "linear_attn.in_proj_b.weight"]
                out_sc, out_q = qweights[p + "linear_attn.out_proj.weight"]

                mixed = qmatmul(qkv_sc, qkv_q, xb)
                conv_w = weights[p + "linear_attn.conv1d.weight"].reshape(cfg.linear_conv_dim, cfg.linear_conv_kernel)
                state = conv_state[layer]
                state[:, :-1] = state[:, 1:]
                state[:, -1] = mixed
                mixed = silu_approx(np.sum(state * conv_w, axis=1))

                query = mixed[: cfg.linear_key_dim].reshape(cfg.linear_num_k_heads, cfg.linear_key_head_dim)
                key = mixed[cfg.linear_key_dim : 2 * cfg.linear_key_dim].reshape(cfg.linear_num_k_heads, cfg.linear_key_head_dim)
                value = mixed[2 * cfg.linear_key_dim :].reshape(cfg.linear_num_v_heads, cfg.linear_value_head_dim)
                z = qmatmul(z_sc, z_q, xb).reshape(cfg.linear_num_v_heads, cfg.linear_value_head_dim)
                a = qmatmul(a_sc, a_q, xb)
                b = qmatmul(b_sc, b_q, xb)

                beta = sigmoid_approx(b)
                g = -exp_approx(weights[p + "linear_attn.A_log"]) * softplus_approx(a + weights[p + "linear_attn.dt_bias"])
                core = np.zeros((cfg.linear_num_v_heads, cfg.linear_value_head_dim), dtype=np.float32)
                head_rep = cfg.linear_num_v_heads // cfg.linear_num_k_heads
                for h in range(cfg.linear_num_v_heads):
                    src_head = h // head_rep
                    qh = l2norm(query[src_head]) * np.float32(1.0 / np.sqrt(cfg.linear_key_head_dim))
                    kh = l2norm(key[src_head])
                    hs = recurrent_state[layer, h]
                    hs *= exp_approx(g[h])
                    kv_mem = np.sum(hs * kh[:, None], axis=0)
                    delta = (value[h] - kv_mem) * beta[h]
                    hs += kh[:, None] * delta[None, :]
                    core[h] = np.sum(hs * qh[:, None], axis=0)

                norm_weight = weights[p + "linear_attn.norm.weight"].astype(np.float32)
                for h in range(cfg.linear_num_v_heads):
                    core[h] = rmsnorm(core[h], norm_weight, cfg.rms_eps) * silu_approx(z[h])

                x = x + qmatmul(out_sc, out_q, core.reshape(cfg.linear_value_dim))
            else:
                raise ValueError(f"unsupported layer type {layer_type}")

            xb2 = rmsnorm(x, qwen_rms_weight(weights[p + "post_attention_layernorm.weight"]), cfg.rms_eps)
            g_sc, g_q = qweights[p + "mlp.gate_proj.weight"]
            u_sc, u_q = qweights[p + "mlp.up_proj.weight"]
            d_sc, d_q = qweights[p + "mlp.down_proj.weight"]
            gate = qmatmul(g_sc, g_q, xb2)
            up = qmatmul(u_sc, u_q, xb2)
            hidden = silu_approx(gate) * up
            x = x + qmatmul(d_sc, d_q, hidden)

        x = rmsnorm(x, final_norm, cfg.rms_eps)
        next_tok = chunked_argmax_embed(emb_scales, emb_q, x)
        if pos >= len(tokens) - 1:
            out.append(next_tok)

    if predict <= 0:
        return []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Qwen3.5 text-only blobs for the teaching kernel")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--predict", type=int, default=1)
    ap.add_argument("--runtime-seq-len", type=int, default=DEFAULT_RUNTIME_SEQ)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    cfg = load_config(model_dir)
    runtime_seq_len = min(cfg.seq_len, args.runtime_seq_len)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("prompt produced no tokens")
    if len(prompt_tokens) + args.predict > runtime_seq_len:
        raise ValueError("prompt + predict exceeds runtime_seq_len")

    weights, qweights, emb_scales, emb_q, final_norm = build_runtime_weights(model_dir, cfg)
    expected = forward_quantized(cfg, weights, qweights, emb_scales, emb_q, final_norm, prompt_tokens, args.predict)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_blob = pack_qmat(emb_scales, emb_q)
    (out_dir / "EMB.BIN").write_bytes(emb_blob)
    (out_dir / "NRM.BIN").write_bytes(np.asarray(final_norm, dtype="<f4").tobytes())
    for layer, layer_type in enumerate(cfg.layer_types):
        if layer_type == "full_attention":
            blob = blob_full_layer(weights, qweights, layer)
        elif layer_type == "linear_attention":
            blob = blob_linear_layer(weights, qweights, layer)
        else:
            raise ValueError(f"unsupported layer type {layer_type}")
        (out_dir / f"L{layer:02d}.BIN").write_bytes(blob)

    write_cfg_file(out_dir / "CFG.BIN", cfg, runtime_seq_len)
    write_layer_types(out_dir / "LTY.BIN", cfg)
    write_u32_file(out_dir / "PMT.BIN", prompt_tokens)
    write_u32_file(out_dir / "EXP.BIN", expected)
    write_rope_file(out_dir / "ROP.BIN", cfg, runtime_seq_len)
    write_text_sidecars(out_dir, tokenizer, args.prompt, prompt_tokens, expected)
    (out_dir / "LAYERS.TXT").write_text(
        "".join(f"{idx:02d} {layer_type}\n" for idx, layer_type in enumerate(cfg.layer_types))
    )

    info = {
        "model": str(model_dir),
        "model_kind": "qwen3.5-text-only",
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "expected_tokens": expected,
        "decoded_expected": tokenizer.decode(expected),
        "config": {
            "dim": cfg.dim,
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
            "runtime_seq_len": runtime_seq_len,
            "rope_theta": cfg.rope_theta,
            "rope_rotary_dim": cfg.rope_rotary_dim,
            "linear_num_k_heads": cfg.linear_num_k_heads,
            "linear_num_v_heads": cfg.linear_num_v_heads,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "linear_conv_kernel": cfg.linear_conv_kernel,
        },
        "layer_mix": {
            "full_attention": sum(1 for t in cfg.layer_types if t == "full_attention"),
            "linear_attention": sum(1 for t in cfg.layer_types if t == "linear_attention"),
        },
        "files": {
            "cfg_bytes": struct.calcsize("<13I2f6I"),
            "embedding_bytes": len(emb_blob),
            "layer0_bytes": (out_dir / "L00.BIN").stat().st_size,
        },
    }
    (out_dir / "INFO.TXT").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
