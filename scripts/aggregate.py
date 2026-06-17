#!/usr/bin/env python3
"""Aggregate all benchmark result JSONs into one grouped table."""
import glob, json, os

LABELS = {
    # 0.6B mbs8
    "vanilla": "0.6B vanilla SDPA mbs8",
    "vanilla_flash": "0.6B vanilla flash mbs8",
    "optimized": "0.6B optimized SDPA mbs8",
    "opt_flash": "0.6B optimized flash mbs8",
    "drop_flash": "0.6B dropout0.05+Liger flash mbs8",
    "opt16_flash": "0.6B optimized 16bit flash mbs8",
    "opt_flash_adamwfused": "0.6B optimized flash adamw_fused mbs8",
    "unsloth": "0.6B UNSLOTH mbs8",
    "unsloth16": "0.6B UNSLOTH 16bit mbs8",
    # 0.6B batch scaling (flash)
    "opt_flash_mbs16": "0.6B optimized flash mbs16",
    "opt_flash_mbs32": "0.6B optimized flash mbs32",
    "opt_flash_mbs48": "0.6B optimized flash mbs48",
    "unsloth_mbs16": "0.6B UNSLOTH mbs16",
    "unsloth_mbs32": "0.6B UNSLOTH mbs32",
    # 1.7B
    "axo17b": "1.7B optimized SDPA mbs4",
    "axo17b_flash": "1.7B optimized flash mbs4",
    "axo17b_flash_mbs8": "1.7B optimized flash mbs8",
    "axo17b_flash_mbs16": "1.7B optimized flash mbs16",
    "unsloth17b": "1.7B UNSLOTH mbs4",
    "unsloth17b_mbs8": "1.7B UNSLOTH mbs8",
    "unsloth17b_mbs16": "1.7B UNSLOTH mbs16",
}
ORDER = list(LABELS.keys())
rows = []
for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "*.json")):
    tag = os.path.splitext(os.path.basename(f))[0]
    if tag.startswith("_"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not d.get("train_runtime"):
        continue
    rows.append({"tag": tag, "label": LABELS.get(tag, tag),
                 "tok_s": d.get("total_tok_per_s"), "mem": d.get("peak_reserved_gib")})
rows.sort(key=lambda r: ORDER.index(r["tag"]) if r["tag"] in ORDER else 999)
w = max((len(r["label"]) for r in rows), default=10)
print(f"{'config':<{w}}  {'tok/s':>8}  {'mem_GiB':>7}")
print("-" * (w + 20))
for r in rows:
    print(f"{r['label']:<{w}}  {str(r['tok_s']):>8}  {str(r['mem']):>7}")
