#!/usr/bin/env python3
"""
Export Qwen bonus-only metadata assets.

This script does not export runnable Qwen weights. It emits only model
configuration and layer-type metadata so the Qwen bonus analysis path stays
clearly separated from the runnable SmolLM mainline and the heavier
text-only Qwen exporter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

CFG_MAGIC = 0x31474643
CFG_VERSION = 1
MAX_RUNTIME_SEQ = 256

LAYER_FULL = 0
LAYER_LINEAR = 1
LAYER_UNKNOWN = 2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_cfg(path: Path, text_cfg: dict) -> None:
    seq_len = int(text_cfg.get("max_position_embeddings", 0))
    runtime_seq_len = min(seq_len, MAX_RUNTIME_SEQ) if seq_len > 0 else MAX_RUNTIME_SEQ
    raw = struct.pack(
        "<10I2f",
        CFG_MAGIC,
        CFG_VERSION,
        int(text_cfg.get("hidden_size", 0)),
        int(text_cfg.get("intermediate_size", 0)),
        int(text_cfg.get("num_hidden_layers", 0)),
        int(text_cfg.get("num_attention_heads", 0)),
        int(text_cfg.get("num_key_value_heads", 0)),
        int(text_cfg.get("vocab_size", 0)),
        seq_len,
        runtime_seq_len,
        float(text_cfg.get("rms_norm_eps", 0.0)),
        float(text_cfg.get("rope_theta", 0.0)),
    )
    path.write_bytes(raw)


def layer_code(layer_type: str) -> int:
    if layer_type == "full_attention":
        return LAYER_FULL
    if layer_type == "linear_attention":
        return LAYER_LINEAR
    return LAYER_UNKNOWN


def write_layer_types(path: Path, layer_types: list[str]) -> None:
    raw = struct.pack("<I", len(layer_types))
    if layer_types:
        raw += struct.pack("<" + "I" * len(layer_types), *(layer_code(t) for t in layer_types))
    path.write_bytes(raw)


def layer_summary(layer_types: list[str]) -> dict[str, int]:
    return {
        "linear_attention": sum(1 for t in layer_types if t == "linear_attention"),
        "full_attention": sum(1 for t in layer_types if t == "full_attention"),
        "unknown": sum(1 for t in layer_types if t not in ("linear_attention", "full_attention")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Qwen bonus-only metadata assets")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    cfg = read_json(model_dir / "config.json")
    index = read_json(model_dir / "model.safetensors.index.json")
    text_cfg = cfg.get("text_config", {})
    weight_map: dict[str, str] = index.get("weight_map", {})
    keys = sorted(weight_map)
    layer_types = list(text_cfg.get("layer_types", []))

    has_vision = any(k.startswith("model.visual.") for k in keys)
    has_mtp = any(k.startswith("mtp.") for k in keys)
    blockers: list[str] = []
    if has_vision:
        blockers.append("model contains vision weights and the current guest path still runs text-only")
    if has_mtp:
        blockers.append("model contains MTP weights and the guest runtime still ignores them")
    blockers.append("the runnable text-only Qwen path is substantially heavier than SmolLM in guest RAM and load time")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_cfg(out_dir / "CFG.BIN", text_cfg)
    write_layer_types(out_dir / "LTY.BIN", layer_types)
    (out_dir / "LAYERS.TXT").write_text(
        "".join(f"{idx:02d} {layer_type}\n" for idx, layer_type in enumerate(layer_types))
    )

    info = {
        "model": str(model_dir),
        "bonus_only": True,
        "runnable_in_guest": False,
        "runnable_text_only_path_exists": True,
        "exported_files": ["CFG.BIN", "LTY.BIN", "LAYERS.TXT", "INFO.TXT"],
        "completed_prereqs": [
            "CFG.BIN-compatible core dimension export exists",
            "a separate text-only Qwen exporter/runtime path now exists via tools/export_qwen.py and make qwen-assets",
        ],
        "text_config": {
            "hidden_size": text_cfg.get("hidden_size"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "num_attention_heads": text_cfg.get("num_attention_heads"),
            "num_key_value_heads": text_cfg.get("num_key_value_heads"),
            "head_dim": text_cfg.get("head_dim"),
            "vocab_size": text_cfg.get("vocab_size"),
            "max_position_embeddings": text_cfg.get("max_position_embeddings"),
            "rope_theta": text_cfg.get("rope_theta"),
            "rope_parameters": text_cfg.get("rope_parameters"),
        },
        "layer_mix": layer_summary(layer_types),
        "weights": {
            "total_tensors": len(keys),
            "has_vision": has_vision,
            "has_mtp": has_mtp,
        },
        "blockers": blockers,
        "next_steps": [
            "use this metadata-only export to teach students how to inspect the mixed layer stack before reading runnable blobs",
            "compare this metadata-only view against the runnable text-only exporter output from tools/export_qwen.py",
            "after Lab 2/3, revisit mmap/COW/buffer-cache improvements for the runnable Qwen path",
        ],
    }
    text = json.dumps(info, indent=2) + "\n"
    (out_dir / "INFO.TXT").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
