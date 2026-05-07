#!/usr/bin/env python3
"""
Export SmolLM2-135M-Instruct into a simple row-wise int8 format that the
teaching kernel can consume from many small root-directory files.

The runtime intentionally targets a single-model, single-batch, short-context
bring-up path. It is not a general-purpose converter.

This exporter is the required/mainline course path. Its output format is kept
deliberately simple so students can inspect every artifact on the guest side.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

CFG_MAGIC = 0x31474643
CFG_VERSION = 1
MAX_RUNTIME_SEQ = 256


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    rms_eps: float
    rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_kv_heads

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads


def load_config(model_dir: Path) -> ModelConfig:
    raw = json.loads((model_dir / "config.json").read_text())
    if raw.get("model_type") != "llama":
        raise ValueError("only llama-like SmolLM exports are supported")
    return ModelConfig(
        dim=int(raw["hidden_size"]),
        hidden_dim=int(raw["intermediate_size"]),
        n_layers=int(raw["num_hidden_layers"]),
        n_heads=int(raw["num_attention_heads"]),
        n_kv_heads=int(raw["num_key_value_heads"]),
        vocab_size=int(raw["vocab_size"]),
        seq_len=int(raw["max_position_embeddings"]),
        rms_eps=float(raw["rms_norm_eps"]),
        rope_theta=float(raw.get("rope_theta", 10000.0)),
    )


def tensor_to_f32(sf: safe_open, name: str) -> np.ndarray:
    return sf.get_tensor(name).to(dtype=torch.float32).cpu().numpy()


def rowwise_q8(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mat.ndim != 2:
        raise ValueError("expected rank-2 tensor")
    a = np.asarray(mat, dtype=np.float32, order="C")
    maxabs = np.max(np.abs(a), axis=1)
    scales = np.where(maxabs > 0.0, maxabs / 127.0, 1.0).astype(np.float32)
    q = np.rint(a / scales[:, None]).clip(-127, 127).astype(np.int8)
    return scales, q


def pack_qmat(scales: np.ndarray, q: np.ndarray) -> bytes:
    return scales.astype("<f4", copy=False).tobytes() + q.astype(np.int8, copy=False).tobytes()


def poly_exp_pos(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 8.0).astype(np.float32)
    return 1.0 + x * (
        1.0
        + x * (
            0.5
            + x * (
                (1.0 / 6.0)
                + x * ((1.0 / 24.0) + x * (1.0 / 120.0))
            )
        )
    )


def exp_approx(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = poly_exp_pos(x[pos])
    out[~pos] = 1.0 / poly_exp_pos(-x[~pos])
    return out


def sigmoid_approx(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + exp_approx(-x))


def silu_approx(x: np.ndarray) -> np.ndarray:
    return x * sigmoid_approx(x)


def softmax_approx(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x)
    y = exp_approx(np.clip(x - m, -8.0, 0.0))
    return y / np.sum(y)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    inv = np.float32(1.0 / math.sqrt(float(np.mean(x * x) + eps)))
    return (x * inv) * weight


def rope_angles(pos: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    idx = np.arange(half, dtype=np.float32)
    freqs = pos / np.power(theta, (2.0 * idx) / head_dim)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def apply_rope_inplace(vec: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> None:
    half = vec.shape[-1] // 2
    x1 = vec[:half].copy()
    x2 = vec[half:].copy()
    vec[:half] = x1 * cos - x2 * sin
    vec[half:] = x1 * sin + x2 * cos


def qmatmul(scales: np.ndarray, q: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = q.astype(np.float32) @ x
    y *= scales
    return y.astype(np.float32, copy=False)


def stream_argmax_embed(scales: np.ndarray, q: np.ndarray, x: np.ndarray) -> int:
    best_idx = 0
    best_val = -1.0e30
    for i in range(q.shape[0]):
        v = float(np.dot(q[i].astype(np.float32), x) * scales[i])
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def blob_layer(weights: dict[str, np.ndarray], qweights: dict[str, tuple[np.ndarray, np.ndarray]], layer: int, cfg: ModelConfig) -> bytes:
    p = f"model.layers.{layer}."
    parts: list[bytes] = []

    def add_vec(name: str) -> None:
        parts.append(np.asarray(weights[p + name], dtype="<f4").tobytes())

    def add_qmat(name: str) -> None:
        scales, q = qweights[p + name]
        parts.append(pack_qmat(scales, q))

    add_vec("input_layernorm.weight")
    add_vec("post_attention_layernorm.weight")
    add_qmat("self_attn.q_proj.weight")
    add_qmat("self_attn.k_proj.weight")
    add_qmat("self_attn.v_proj.weight")
    add_qmat("self_attn.o_proj.weight")
    add_qmat("mlp.gate_proj.weight")
    add_qmat("mlp.up_proj.weight")
    add_qmat("mlp.down_proj.weight")
    return b"".join(parts)


def build_runtime_weights(
    model_dir: Path, cfg: ModelConfig
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(safetensors_path)

    wanted: list[str] = ["model.embed_tokens.weight", "model.norm.weight"]
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        wanted.extend(
            [
                p + "input_layernorm.weight",
                p + "post_attention_layernorm.weight",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
        )

    weights: dict[str, np.ndarray] = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as sf:
        for name in wanted:
            weights[name] = tensor_to_f32(sf, name)

    qweights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, arr in weights.items():
        if arr.ndim == 2:
            qweights[name] = rowwise_q8(arr)

    emb_scales, emb_q = qweights["model.embed_tokens.weight"]
    final_norm = np.asarray(weights["model.norm.weight"], dtype=np.float32)
    return weights, qweights, emb_scales, emb_q, final_norm


def forward_quantized(
    cfg: ModelConfig,
    weights: dict[str, np.ndarray],
    qweights: dict[str, tuple[np.ndarray, np.ndarray]],
    emb_scales: np.ndarray,
    emb_q: np.ndarray,
    final_norm: np.ndarray,
    tokens: list[int],
    predict: int,
) -> list[int]:
    head_dim = cfg.head_dim
    kv_dim = cfg.kv_dim
    kv_mul = cfg.kv_mul
    cache_k = np.zeros((cfg.n_layers, len(tokens) + predict, cfg.n_kv_heads, head_dim), dtype=np.float32)
    cache_v = np.zeros((cfg.n_layers, len(tokens) + predict, cfg.n_kv_heads, head_dim), dtype=np.float32)
    out: list[int] = []

    for pos in range(len(tokens) + predict - 1):
        token = tokens[pos] if pos < len(tokens) else out[pos - len(tokens)]
        x = emb_q[token].astype(np.float32) * emb_scales[token]

        for layer in range(cfg.n_layers):
            p = f"model.layers.{layer}."
            xb = rmsnorm(x, weights[p + "input_layernorm.weight"], cfg.rms_eps)

            q_sc, q_q = qweights[p + "self_attn.q_proj.weight"]
            k_sc, k_q = qweights[p + "self_attn.k_proj.weight"]
            v_sc, v_q = qweights[p + "self_attn.v_proj.weight"]
            o_sc, o_q = qweights[p + "self_attn.o_proj.weight"]
            g_sc, g_q = qweights[p + "mlp.gate_proj.weight"]
            u_sc, u_q = qweights[p + "mlp.up_proj.weight"]
            d_sc, d_q = qweights[p + "mlp.down_proj.weight"]

            qv = qmatmul(q_sc, q_q, xb).reshape(cfg.n_heads, head_dim)
            kv = qmatmul(k_sc, k_q, xb).reshape(cfg.n_kv_heads, head_dim)
            vv = qmatmul(v_sc, v_q, xb).reshape(cfg.n_kv_heads, head_dim)

            cos, sin = rope_angles(pos, head_dim, cfg.rope_theta)
            for h in range(cfg.n_heads):
                apply_rope_inplace(qv[h], cos, sin)
            for h in range(cfg.n_kv_heads):
                apply_rope_inplace(kv[h], cos, sin)

            cache_k[layer, pos] = kv
            cache_v[layer, pos] = vv

            att = np.zeros((cfg.n_heads, head_dim), dtype=np.float32)
            scale = np.float32(1.0 / math.sqrt(head_dim))
            for h in range(cfg.n_heads):
                kv_h = h // kv_mul
                scores = np.zeros(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    scores[t] = np.dot(qv[h], cache_k[layer, t, kv_h]) * scale
                probs = softmax_approx(scores)
                for t in range(pos + 1):
                    att[h] += probs[t] * cache_v[layer, t, kv_h]

            x = x + qmatmul(o_sc, o_q, att.reshape(cfg.dim))

            xb2 = rmsnorm(x, weights[p + "post_attention_layernorm.weight"], cfg.rms_eps)
            gate = qmatmul(g_sc, g_q, xb2)
            up = qmatmul(u_sc, u_q, xb2)
            hidden = silu_approx(gate) * up
            x = x + qmatmul(d_sc, d_q, hidden)

        x = rmsnorm(x, final_norm, cfg.rms_eps)
        next_tok = stream_argmax_embed(emb_scales, emb_q, x)
        if pos >= len(tokens) - 1:
            out.append(next_tok)

    if predict <= 0:
        return []
    return out


def write_u32_file(path: Path, values: list[int]) -> None:
    raw = struct.pack("<I", len(values))
    raw += struct.pack("<" + "I" * len(values), *values)
    path.write_bytes(raw)


def token_text_lines(tokenizer: AutoTokenizer, tokens: list[int]) -> str:
    if not tokens:
        return "(none)\n"
    return "".join(f"{tok}: {json.dumps(tokenizer.decode([tok]))}\n" for tok in tokens)


def write_text_sidecars(out_dir: Path, tokenizer: AutoTokenizer, prompt: str, prompt_tokens: list[int], expected: list[int]) -> None:
    # These text files let students verify that token ids correspond to real,
    # meaningful text without needing a tokenizer inside the teaching kernel.
    prompt_text = (
        f"prompt_text = {json.dumps(prompt)}\n"
        f"prompt_tokens = {prompt_tokens}\n"
        f"decoded_prompt = {json.dumps(tokenizer.decode(prompt_tokens))}\n"
        "prompt_token_texts:\n"
        f"{token_text_lines(tokenizer, prompt_tokens)}"
    )
    expect_text = (
        f"expected_tokens = {expected}\n"
        f"decoded_expected = {json.dumps(tokenizer.decode(expected))}\n"
        "expected_token_texts:\n"
        f"{token_text_lines(tokenizer, expected)}"
    )
    (out_dir / "PROMPT.TXT").write_text(prompt_text)
    (out_dir / "EXPECT.TXT").write_text(expect_text)


def write_cfg_file(path: Path, cfg: ModelConfig) -> None:
    runtime_seq_len = min(cfg.seq_len, MAX_RUNTIME_SEQ)
    raw = struct.pack(
        "<10I2f",
        CFG_MAGIC,
        CFG_VERSION,
        cfg.dim,
        cfg.hidden_dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.vocab_size,
        cfg.seq_len,
        runtime_seq_len,
        cfg.rms_eps,
        cfg.rope_theta,
    )
    path.write_bytes(raw)


def write_rope_file(path: Path, cfg: ModelConfig, max_pos: int) -> None:
    raw = bytearray()
    for pos in range(max_pos):
        cosv, sinv = rope_angles(pos, cfg.head_dim, cfg.rope_theta)
        raw.extend(np.asarray(cosv, dtype="<f4").tobytes())
        raw.extend(np.asarray(sinv, dtype="<f4").tobytes())
    path.write_bytes(bytes(raw))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export SmolLM2 into tiny-kernel-friendly blobs")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--predict", type=int, default=1)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    cfg = load_config(model_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("prompt produced no tokens")

    if len(prompt_tokens) + args.predict > 256:
        raise ValueError("prompt + predict must stay within 256 tokens")

    weights, qweights, emb_scales, emb_q, final_norm = build_runtime_weights(model_dir, cfg)
    expected = forward_quantized(cfg, weights, qweights, emb_scales, emb_q, final_norm, prompt_tokens, args.predict)

    emb_blob = pack_qmat(emb_scales, emb_q)
    (out_dir / "EMB.BIN").write_bytes(emb_blob)
    (out_dir / "NRM.BIN").write_bytes(np.asarray(final_norm, dtype="<f4").tobytes())

    for layer in range(cfg.n_layers):
        blob = blob_layer(weights, qweights, layer, cfg)
        (out_dir / f"L{layer:02d}.BIN").write_bytes(blob)

    write_cfg_file(out_dir / "CFG.BIN", cfg)
    write_u32_file(out_dir / "PMT.BIN", prompt_tokens)
    write_u32_file(out_dir / "EXP.BIN", expected)
    write_rope_file(out_dir / "ROP.BIN", cfg, min(cfg.seq_len, MAX_RUNTIME_SEQ))
    write_text_sidecars(out_dir, tokenizer, args.prompt, prompt_tokens, expected)

    info = {
        "model": str(model_dir),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "expected_tokens": expected,
        "config": {
            "dim": cfg.dim,
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
            "runtime_seq_len": min(cfg.seq_len, MAX_RUNTIME_SEQ),
            "rope_theta": cfg.rope_theta,
            "rms_eps": cfg.rms_eps,
        },
        "files": {
            "cfg_bytes": struct.calcsize("<10I2f"),
            "embedding_bytes": len(emb_blob),
            "layer_blob_bytes": len(blob_layer(weights, qweights, 0, cfg)),
        },
        "decoded_expected": tokenizer.decode(expected),
    }
    (out_dir / "INFO.TXT").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
