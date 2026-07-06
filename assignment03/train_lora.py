"""
train_lora.py — QLoRA SFT for FinQA-VN program generation.

Fine-tunes a <=9B instruct model to emit the DSL program (in the exact
"PROGRAM: ..." inference format). Trains ONLY on the assistant tokens
(prompt is masked with -100), 4-bit base + LoRA adapters. Self-contained on
transformers + peft (no trl) for API stability.

Usage (local or via Modal):
    python3 train_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train data_sft/train_sft.jsonl --eval data_sft/eval_sft.jsonl \
        --output-dir runs/lora_qwen25_7b --epochs 2
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from datasets import load_dataset as hf_load
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train", default="data_sft/train_sft.jsonl")
    p.add_argument("--eval", default="data_sft/eval_sft.jsonl")
    p.add_argument("--output-dir", default="runs/lora_qwen25_7b")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tmpl(messages, add_generation_prompt):
        # Qwen3 is a thinking model: force enable_thinking=False so training targets
        # a clean "PROGRAM: ..." (matching submit.py inference with cot_format=NONE).
        # Qwen2.5 tokenizers don't accept the kwarg -> fall back.
        try:
            return tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            )

    def encode(ex):
        # full conversation and prompt-only, so we can mask the prompt.
        msgs = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        full = _tmpl(msgs, add_generation_prompt=False)
        prompt = _tmpl(msgs[:2], add_generation_prompt=True)
        labels = list(full)
        for i in range(min(len(prompt), len(labels))):
            labels[i] = -100
        # truncate from the LEFT of the prompt so the assistant answer is never cut
        if len(full) > args.max_seq_len:
            overflow = len(full) - args.max_seq_len
            # keep the beginning special tokens? simplest: drop overflow from the middle of the prompt
            full = full[:1] + full[1 + overflow:]
            labels = labels[:1] + labels[1 + overflow:]
        return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}

    ds = hf_load("json", data_files={"train": args.train, "eval": args.eval})
    ds = ds.map(encode, remove_columns=ds["train"].column_names)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    # Make gradient checkpointing actually effective through PEFT (big memory saver).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps", eval_steps=50,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["eval"],
        data_collator=collate,
    )
    trainer.train()
    final = Path(args.output_dir) / "adapter_final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"Saved LoRA adapter to {final}")


if __name__ == "__main__":
    main()
