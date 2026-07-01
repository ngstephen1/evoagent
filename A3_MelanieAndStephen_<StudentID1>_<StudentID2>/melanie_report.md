# Melanie's Phase-3 Work Log — Reaching 0.694 on Kaggle

**Author:** Melanie
**Kaggle ID:** `yeyeezyzeus`
**Best public score achieved this session:** **0.69433** (`submission_ensemble.csv`)
**Previous team best:** 0.65789 (Run009-lite hybrid, 4B model)
**Net improvement:** **+0.0365** (about +3.6 points)

---

## 1. Headline

Starting from the team's existing best of **0.65789** (built on a 4B model with hand-engineered
hybrid patching), I pushed the public score to **0.69433** — close to 70% — by moving to a larger
model, adding self-consistency, and, most importantly, **majority-voting an ensemble of three
diverse models**. The single biggest lever was the ensemble: it beat every individual submission by
~3.5 points.

---

## 2. Starting Point

| Item | Value |
|---|---|
| Team best before this session | 0.65789 (Kaggle public) |
| Model behind it | `QuantTrio/Qwen3.5-4B-AWQ` (4B, AWQ-quantized) |
| Method | Hybrid fallback ensembling + targeted retry (Run003 → Run008 → Run009) |
| Base dev accuracy (4B) | 0.4833 |
| Compute | Originally Modal A10G; this session used **4× NVIDIA H100 80GB** via SSH |

The 4B pipeline had plateaued — successive hybrid/retry runs were adding only ~0.1–0.2 points each.

---

## 3. What I Did — Step by Step (one variable at a time)

I followed the scientific method: **change exactly one variable per run, measure, then decide.**

### Step 1 — Upgrade the model (4B → 8B)
- **Change:** `QuantTrio/Qwen3.5-4B-AWQ` → `Qwen/Qwen3-8B` (full bf16, rules-legal ≤9B), nothing else.
- **Why:** The rules allow open-weight models up to 9B; the team had only used 4B. The 4× H100s made
  an 8B model in full precision easy to run.
- **Result:** **Dev accuracy jumped 0.4833 → 0.650** (+16.7 points). The single biggest quality gain.
- **Multi-GPU:** Added `--dp-size 4` (data parallelism) so the 8B model runs 4 replicas across the
  H100s — ~4× faster, identical results.

### Step 2 — Self-consistency (multi-sample voting)
- **Change:** Sample **k programs per question** at temperature 0.6, execute each, and keep the
  **majority-voted executed value** (instead of one greedy pass). Model held fixed at Qwen3-8B.
- **Why:** At temperature 0 the model commits to one program; if that single attempt slips (wrong row,
  wrong operation) it's wrong. Voting over several samples outvotes one-off slips.
- **Results:**
  - k=5 → **dev 0.704** (cleared 70% on dev!), Kaggle 0.64979
  - k=16 → Kaggle **0.65587** (zeros dropped 15 → 11)
- **Note:** Dev hit 0.704 but Kaggle stayed ~0.65 — dev and test distributions differ, and the old
  hybrid still edged single strategies via failure-patching.

### Step 3 — A different model, for diversity
- **Change:** Generated a submission with `Qwen/Qwen2.5-Coder-7B-Instruct` (+ self-consistency k=8).
- **Why:** Not to win alone, but to feed a **diverse third opinion** into an ensemble. A code-tuned
  model produces cleaner DSL syntax and *different* mistakes than Qwen3.
- **Result:** **0.48178 alone** (weak — worse Vietnamese comprehension), but only 8 zero rows (cleanest
  DSL). Its value was its *disagreement* with the other models, not its standalone score.

### Step 4 — 3-way majority-vote ensemble  ← **the breakthrough**
- **Change:** For every test row, majority-vote across three submissions:
  1. Qwen3-8B + self-consistency k=16 (0.65587)
  2. Old team hybrid (0.65789)
  3. Qwen2.5-Coder-7B (0.48178)
  Ties broken toward the old hybrid (the safest prior).
- **Why:** The three strategies **agreed on 65% of rows and disagreed on ~175**. On the disagreement
  rows, a majority vote of independent models breaks ties toward the more-likely-correct answer.
- **Result:** **0.69433** — beating every individual submission by ~3.5 points. 70 rows were changed
  from the old best, and the net effect was strongly positive.

---

## 4. The Score Journey

| # | Submission | Model / Method | Dev | Kaggle Public |
|---|---|---|---:|---:|
| — | (team baseline) | 4B hybrid (Run009) | 0.483 | 0.65789 |
| 1 | 8B greedy | Qwen3-8B | **0.650** | — |
| 2 | 8B + SC k=5 | + self-consistency | **0.704** | 0.64979 |
| 3 | 8B + SC k=16 | more voting | — | 0.65587 |
| 4 | Coder-7B | Qwen2.5-Coder-7B (alone) | — | 0.48178 |
| 5 | **3-way ensemble** | **majority vote of #3 + baseline + #4** | — | **0.69433** 🏆 |

---

## 5. What Actually Helped (and What Didn't)

**Helped the most:**
1. **The diverse ensemble (+3.5 pts).** By far the biggest Kaggle gain. Combining models that make
   *different* mistakes and majority-voting is stronger than any single model.
2. **Bigger model, 4B → 8B (+16.7 pts on dev).** The largest quality gain to the base solver.
3. **Self-consistency.** Cleared 70% on dev and reduced zero-prediction failures (15 → 8–11).

**Key counter-intuitive lesson:**
- **A weak model can strengthen an ensemble.** Coder-7B scored only 0.48 alone, yet it lifted the
  ensemble to 0.694 — because its *diversity* was more valuable than its standalone accuracy.

**What didn't move the needle:**
- Higher self-consistency `k` (5 → 16) gave only ~+0.6 on Kaggle — the `k` lever saturates.
- A single strong model, no matter how good on dev, did **not** automatically beat the engineered
  hybrid on Kaggle (dev ↔ test distribution gap).

---

## 6. Technical Details

- **Models (all rules-compliant ≤9B open-weight):** `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-Coder-7B-Instruct`.
- **Inference:** SGLang, `dtype=bfloat16`, data-parallel across 4× H100 (`--dp-size 4`).
- **Self-consistency:** k samples at temp 0.6 → execute each program with the local DSL evaluator →
  keep the majority-voted executed value.
- **Ensemble:** custom `phase3_ensemble_vote.py` — per-row numeric majority vote across N submission
  CSVs with a priority tiebreaker (CPU-only, runs locally).
- **Integrity:** no hidden labels, no manual test labeling. All predictions came from documented model
  inference and deterministic ensemble rules. HF tokens / keys kept out of the repo.

---

## 7. Reproducibility (key commands)

```bash
# Step 1+2: evolve strategy with 8B + self-consistency (on the GPU box)
python3 arc_proofs.py evolution --T 5 --train-size 200 --dev-size 240 \
  --output-dir runs/exp_qwen3_8b_sc5 --model Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.9 --dp-size 4 --self-consistency-k 5

# Step 3: generate the 8B (k=16) and Coder-7B submissions
python3 submit.py --strategy-path runs/exp_qwen3_8b_sc5/iter_best_strategy.json \
  --output-file runs/kaggle_8b_sc16/submission.csv --model Qwen/Qwen3-8B \
  --dp-size 4 --self-consistency-k 16
python3 submit.py --strategy-path runs/exp_qwen3_8b_sc5/iter_best_strategy.json \
  --output-file runs/kaggle_coder7b/submission.csv --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dp-size 4 --self-consistency-k 8

# Step 4: 3-way majority-vote ensemble (runs locally, CPU only)
python3 phase3_ensemble_vote.py \
  --inputs submission_8b_sc16.csv final_submission.csv submission_coder7b.csv \
  --priority final_submission.csv \
  --output submission_ensemble.csv
```

Code added this session: `--tp-size`/`--dp-size`/`--self-consistency-k` flags in `model.py`,
`main.py`, `submit.py`, `arc_proofs.py`; self-consistency voting in `executor.py`; and the new
`phase3_ensemble_vote.py`.

---

## 8. Next Steps / Honest Limitations

- The remaining gap to 0.80 is large; the base solver caps in the high-0.60s on this test set.
  Reaching much higher would likely need **LoRA fine-tuning** on the train programs or
  **context/table-retrieval compression** to fix wrong-row extraction — both bigger jobs.
- Public ≠ private: the 0.69433 is a **public** score; the final grade is the private leaderboard, so
  the ensemble should be kept as a final candidate alongside the stable old hybrid.
- Adding a 4th diverse model (e.g., Llama-3.1-8B) to the vote could push the ensemble further.

**Final recommendation:** submit `submission_ensemble.csv` (0.69433) as the primary final candidate.
