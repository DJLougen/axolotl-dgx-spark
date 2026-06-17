#!/usr/bin/env python3
"""Unsloth reference benchmark: Qwen3-0.6B QLoRA, identical shape to the Axolotl runs.

Fixed pad-to-1024 pre-tokenized dataset => total token compute matches Axolotl exactly
(30*8*1024), so total_tok/s is directly comparable. Plain HF Trainer drives the
Unsloth-patched model. Writes JSON in the same schema as bench.py.
"""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")

import torch
from unsloth import FastLanguageModel  # must import before transformers bits


def build_dataset(tok, path, seq_len):
    import json as _json
    from datasets import Dataset
    rows = []
    for line in open(path):
        msgs = _json.loads(line)["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tok(text, max_length=seq_len, truncation=True, padding="max_length")
        ids = enc["input_ids"]
        labels = [t if m == 1 else -100 for t, m in zip(ids, enc["attention_mask"])]
        rows.append({"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--mbs", type=int, default=8)
    ap.add_argument("--grad-ckpt", default="false")  # "false" | "unsloth" | "true"
    ap.add_argument("--data", default="data/bench_chat.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="unsloth")
    ap.add_argument("--load-4bit", default="true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    gc = {"false": False, "true": True, "unsloth": "unsloth"}[args.grad_ckpt]
    load_4bit = args.load_4bit.lower() == "true"

    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.seq_len, load_in_4bit=load_4bit, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=gc, random_state=1234,
    )

    ds = build_dataset(tok, args.data, args.seq_len)

    from transformers import Trainer, TrainingArguments, default_data_collator
    targs = TrainingArguments(
        output_dir="outputs/unsloth", per_device_train_batch_size=args.mbs,
        gradient_accumulation_steps=1, max_steps=args.max_steps, warmup_steps=5,
        learning_rate=2e-4, lr_scheduler_type="cosine", weight_decay=0.0,
        optim="adamw_bnb_8bit", bf16=True, tf32=True, logging_steps=1,
        save_strategy="no", report_to=[], seed=1234, dataloader_num_workers=2,
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=default_data_collator)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    wall = time.time() - t0

    rt = out.metrics.get("train_runtime", wall)
    total_tokens = args.max_steps * args.mbs * args.seq_len
    peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
    res = {
        "tag": args.tag, "framework": "unsloth", "measure_steps": args.max_steps,
        "train_runtime": round(rt, 2), "peak_reserved_gib": round(peak, 2),
        "total_tokens": total_tokens, "final_loss": round(out.metrics.get("train_loss", 0), 4),
        "total_tok_per_s": round(total_tokens / rt, 1), "steps_per_s": round(args.max_steps / rt, 4),
        "grad_ckpt": args.grad_ckpt, "seq_len": args.seq_len, "mbs": args.mbs, "load_4bit": load_4bit,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT:", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
