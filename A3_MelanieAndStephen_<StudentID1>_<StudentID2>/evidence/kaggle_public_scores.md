# Kaggle Public Score Evidence

These are the public leaderboard scores recorded during Phase 3. The final
ThinkFlic package uses `submission_ensemble_api3.csv`.

| Submitted file | Public score | Notes |
|---|---:|---|
| `submission_ensemble_api3.csv` | 0.80161 | Primary final; RS Qwen3-8B + Gemini 2.5 Flash + DeepSeek-V3 deterministic ensemble. |
| `submission_ft_qwen3_8b_rs.csv` | 0.74696 | Best self-hosted <=9B alternate; rejection-sampled/STaR Qwen3-8B. |
| `submission_ft_merged.csv` | 0.72874 | Fine-tuned merge with broken-row fallback. |
| `submission_ft_qwen3_8b.csv` | 0.72267 | Fine-tuned Qwen3-8B baseline. |
| `submission_ft_qwen25_7b.csv` | 0.71255 | Fine-tuned Qwen2.5-7B diversity model. |
| `submission_ensemble_v4.csv` | 0.70242 | Best pre-fine-tune team ensemble. |
| `submission_ensemble_v2.csv` | 0.69838 | Earlier member-confirmed post-processing ensemble. |
| `submission_ensemble.csv` | 0.69433 | First team ensemble improvement. |
| `submission_checked.csv` | 0.66194 | GPT-OSS safe 3-change hybrid. |
| `submission_checked.csv` | 0.65789 | Run009-lite safe targeted retry. |
| `submission_checked.csv` | 0.65587 | Run008 filtered targeted retry. |
| `submission_checked.csv` | 0.64574 | Run003/Run004 fallback hybrid plateau. |
| `submission_checked.csv` | 0.56477 | Run001 EvoAgent ARC best strategy baseline. |

Private score and rank remain TBD until the private leaderboard is released.
