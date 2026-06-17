#!/usr/bin/env python3
"""Generate a fixed, deterministic chat dataset for throughput benchmarking.

Each example is a short user prompt + a long assistant answer sized so that,
tokenized with the Qwen3 template, sequences are ~`--target-tokens` long.
This keeps padding low at seq_len=1024 (representative; not a padding benchmark)
while staying fully deterministic so every framework sees identical tokens.
"""
import argparse
import json
import random

TOPICS = [
    "binary search", "gradient descent", "the TCP handshake", "B-trees", "the Krebs cycle",
    "Kalman filters", "the RAFT consensus protocol", "the Fourier transform", "garbage collection",
    "LoRA fine-tuning", "rotary position embeddings", "mixture-of-experts routing",
    "flash attention", "post-training quantization", "the bias-variance tradeoff", "backpropagation",
    "CUDA streams", "virtual memory page tables", "the actor model", "vector clocks",
]
VERBS = ["Explain", "Describe", "Summarize", "Walk me through", "Give an overview of",
         "Outline", "Clarify", "Detail"]
SENTENCES = [
    "The core idea relies on partitioning the problem into smaller subproblems that can be solved independently and then recombined.",
    "Each step refines the current estimate by moving in the direction that reduces the objective the most, scaled by a learning rate.",
    "Memory locality matters because cache misses dominate the runtime cost on modern hardware with deep memory hierarchies.",
    "An invariant must hold before and after every operation, which is what lets us reason about correctness inductively.",
    "In practice the constant factors hidden by asymptotic notation often determine real-world speed more than the exponent does.",
    "We trade additional compute for a reduction in peak memory, which is frequently the binding constraint on accelerators.",
    "The gradient flows backward through each layer, accumulating contributions via repeated application of the chain rule.",
    "Numerical stability is preserved by computing partial results in higher precision and rounding only once at the very end.",
    "Throughput improves when we keep the accelerator busy with large contiguous work and avoid host-device synchronization stalls.",
    "Fusing adjacent operations removes intermediate tensors and the expensive round trips to global memory they would require.",
    "A well-chosen batch size balances parallel hardware utilization against the variance of the stochastic gradient estimate.",
    "Kernel launch overhead is amortized across the many elements processed within a single fused pass over the data.",
    "The method degrades gracefully under noise because each component contributes a bounded, independent share of the error.",
    "Profiling first is essential, since intuition about bottlenecks is unreliable once several subsystems interact.",
    "Correctness and performance are not in tension here: the faster path is also the one that touches less memory.",
    "We cache the expensive transformation once and reuse it, turning repeated quadratic work into a single linear pass.",
]


def make_answer(rng: random.Random, target_sentences: int) -> str:
    # vary +/- a few sentences so lengths differ slightly but stay near target
    n = max(8, target_sentences + rng.randint(-4, 4))
    return " ".join(rng.choice(SENTENCES) for _ in range(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sentences", type=int, default=55,
                    help="approx sentences per answer (~14 tok each -> ~770 tok)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w") as f:
        for _ in range(args.n):
            topic = rng.choice(TOPICS)
            verb = rng.choice(VERBS)
            user = f"{verb} {topic} for a senior software engineer, with concrete reasoning."
            turns = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": make_answer(rng, args.sentences)},
            ]
            f.write(json.dumps({"messages": turns}) + "\n")
    print(f"wrote {args.n} examples to {args.out}")


if __name__ == "__main__":
    main()
