#!/usr/bin/env python3
"""Run one Axolotl benchmark config: warm pass (populate Triton cache) + measured pass.

Parses the HF Trainer summary + per-step memory logs, computes throughput, writes JSON.
Single-process (python -m axolotl.cli.train) so it is trivially killable; wrapped by an
outer `timeout` at launch time for memory-safety on GB10's shared unified memory.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time


def run(cfg: str, max_steps: int, timeout_s: int, extra=None) -> str:
    cmd = [sys.executable, "-m", "axolotl.cli.train", cfg, f"--max-steps={max_steps}"]
    if extra:
        cmd += extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return (e.stdout or "") + (e.stderr or "") + "\n__TIMEOUT__\n"
    return p.stdout + p.stderr


def fnum(pattern: str, text: str, cast=float, reducer=None):
    vals = re.findall(pattern, text)
    if not vals:
        return None
    vals = [cast(v) for v in vals]
    if reducer == "max":
        return max(vals)
    return vals[-1]


def parse(out: str) -> dict:
    return {
        "train_runtime": fnum(r"'train_runtime':\s*'?([0-9.]+)'?", out),
        "samples_per_s_logged": fnum(r"'train_samples_per_second':\s*'?([0-9.]+)'?", out),
        "steps_per_s_logged": fnum(r"'train_steps_per_second':\s*'?([0-9.]+)'?", out),
        "peak_reserved_gib": fnum(r"'memory/device_reserved \(GiB\)':\s*'?([0-9.]+)'?", out, float, "max"),
        "total_tokens": fnum(r"'tokens/total':\s*'?([0-9]+)'?", out, int, "max"),
        "trainable_tokens": fnum(r"'tokens/trainable':\s*'?([0-9]+)'?", out, int, "max"),
        "final_loss": fnum(r"'train_loss':\s*'?([0-9.]+)'?", out),
        "timed_out": "__TIMEOUT__" in out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--warm-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--extra", type=str, default="",
                    help="extra axolotl CLI flags as one string, e.g. \"--torch-compile=true --logging-steps=50\"")
    args = ap.parse_args()

    extra = args.extra.split() if args.extra else None
    print(f"[{args.tag}] WARM pass ({args.warm_steps} steps) to populate caches...", flush=True)
    run(args.config, args.warm_steps, args.timeout, extra)

    print(f"[{args.tag}] MEASURED pass ({args.measure_steps} steps)...", flush=True)
    t0 = time.time()
    out = run(args.config, args.measure_steps, args.timeout, extra)
    wall = time.time() - t0

    res = parse(out)
    res["tag"] = args.tag
    res["config"] = args.config
    res["measure_steps"] = args.measure_steps
    res["wall_including_load_s"] = round(wall, 2)
    rt = res.get("train_runtime")
    # Fall back to deterministic fixed-pad token count if per-step logs were suppressed.
    if not res.get("total_tokens"):
        res["total_tokens"] = args.measure_steps * args.mbs * args.seq
        res["total_tokens_from_shape"] = True
    if rt and res.get("total_tokens"):
        res["total_tok_per_s"] = round(res["total_tokens"] / rt, 1)
        res["steps_per_s"] = round(args.measure_steps / rt, 4)
    if rt and res.get("trainable_tokens"):
        res["trainable_tok_per_s"] = round(res["trainable_tokens"] / rt, 1)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{args.tag}] RESULT: {json.dumps(res)}", flush=True)
    # On failure, dump tail of output for debugging
    if not rt:
        print(f"[{args.tag}] NO train_runtime parsed. Tail of output:\n" + "\n".join(out.splitlines()[-40:]), flush=True)


if __name__ == "__main__":
    main()
