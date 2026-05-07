#!/usr/bin/env python3
"""
Inspect a local Qwen3-family model and report the concrete gaps between it and
the current teaching-kernel runtime.

This does not try to export weights. It reports what remains between the full
multimodal checkpoint and the current teaching-kernel's text-only Qwen bonus
path, so the bonus experiment has a clear systems-oriented scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Qwen3 bonus-model compatibility")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = read_json(model_dir / "config.json")
    index = read_json(model_dir / "model.safetensors.index.json")

    text_cfg = cfg.get("text_config", {})
    weight_map: dict[str, str] = index.get("weight_map", {})
    keys = sorted(weight_map)

    layer_types = list(text_cfg.get("layer_types", []))
    linear_layers = sum(1 for t in layer_types if t == "linear_attention")
    full_layers = sum(1 for t in layer_types if t == "full_attention")
    has_vision = any(k.startswith("model.visual.") for k in keys)
    has_mtp = any(k.startswith("mtp.") for k in keys)
    blockers: list[str] = []
    if has_vision:
        blockers.append("full model contains a vision tower; current guest runtime only exports/runs the text-only language_model path")
    if has_mtp:
        blockers.append("model contains MTP weights; current runtime ignores them and only runs the base text decoder")
    blockers.append("the text-only Qwen path requires a much larger guest RAM budget and noticeably longer load/compute time than SmolLM")

    next_steps = [
        "keep SmolLM as the required path and Qwen as the optional text-only bonus path",
        "when Lab 2 lands, move Qwen blob loading from eager read() toward mmap-backed loading to cut startup copies",
        "when Lab 2 lands, use COW-friendly process structure so the tokenizer/runner service can be shared cheaply",
        "when Lab 3 lands, revisit native-fs read paths plus buffer cache to reduce repeated cold-read cost",
        "only after text-only parity is stable, revisit multimodal vision and MTP weights",
    ]

    report = {
        "model_dir": str(model_dir),
        "architectures": cfg.get("architectures", []),
        "model_type": cfg.get("model_type"),
        "text_config": {
            "hidden_size": text_cfg.get("hidden_size"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "num_attention_heads": text_cfg.get("num_attention_heads"),
            "num_key_value_heads": text_cfg.get("num_key_value_heads"),
            "head_dim": text_cfg.get("head_dim"),
            "vocab_size": text_cfg.get("vocab_size"),
            "max_position_embeddings": text_cfg.get("max_position_embeddings"),
            "rope_parameters": text_cfg.get("rope_parameters"),
        },
        "layer_mix": {
            "linear_attention": linear_layers,
            "full_attention": full_layers,
        },
        "weights": {
            "total_tensors": len(keys),
            "has_vision": has_vision,
            "has_mtp": has_mtp,
        },
        "completed_prereqs": [
            "guest runtime loads model dimensions from CFG.BIN instead of fixed SmolLM constants",
            "guest runtime now supports Qwen text-only layer metadata, q_norm/k_norm, partial RoPE, and linear_attention blocks",
            "host exporter can walk model.language_model.* and emit runnable Qwen text-only blobs",
        ],
        "compatible_with_current_runtime": False,
        "compatible_with_current_text_only_bonus_runtime": True,
        "blockers": blockers,
        "recommended_bonus_scope": "run the text-only Qwen bonus path, then analyze how mmap/COW/native-fs/buffer-cache support would reduce its cost",
        "next_steps": next_steps,
    }

    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
