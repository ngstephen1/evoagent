"""
merge_lora.py — merge a trained LoRA adapter into the base model and save a full
model directory that SGLang can serve directly (via submit.py --model <dir>).

Run in a fresh process AFTER train_lora.py so memory is clean (loads base in bf16,
not 4-bit, to allow a real merge).

Usage:
    python3 merge_lora.py --base Qwen/Qwen2.5-7B-Instruct \
        --adapter runs/lora_qwen25_7b/adapter_final \
        --out runs/lora_merged_qwen25_7b
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, a.adapter)
    merged = merged.merge_and_unload()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    main()
