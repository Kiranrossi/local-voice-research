# Local Voice Models for English and Tamlish

Research comparing **speech-to-text (STT)** and **text-to-speech (TTS)** models that run fully on local hardware, for:

1. **English**: choose a best model and a fallback for STT and TTS, across CPU-only, Apple Silicon (8 GB), and NVIDIA GPU setups.
2. **Tamlish** (Tamil-English code-mixed speech): test whether any open model already handles it, and propose a fine-tuning plan.

## Evaluation approach

Every recommendation rests on six evidence layers:

| # | Layer | What it answers |
|---|-------|-----------------|
| 1 | Published evidence | Leaderboards, model cards, papers |
| 2 | Hands-on reproduction | Does it run, and how fast, on our hardware? |
| 3 | Practical fit | Licence of the weights, quantised builds, runtimes |
| 4 | Indian context | Accuracy on Indian-accented English (Svarah) |
| 5 | Robustness | Hallucination, numbers, names, code-switching failures |
| 6 | Evidence grading | Every claim graded A (independent benchmark) to D (our small test) |

## Hardware profiles

| Profile | Machine |
|---------|---------|
| CPU-only | MacBook Air M2, GPU disabled |
| Apple Silicon, 8 GB | MacBook Air M2 (Metal / MLX) |
| NVIDIA GPU | Google Colab (exact GPU recorded per run) |
| RTX 4090 | Estimated from published figures (not measured) |

## Repository layout

```
notebooks/   Colab and local notebooks (test harness, STT, TTS, Tamlish)
data/        Test sentence lists and manifests (audio files are not committed)
results/     results.csv: one row per model run, the single source for all tables
report/      Final report (PDF) and figures
```

## Reproducing the results

Instructions will be added as notebooks are completed. All numbers in the report are generated from `results/results.csv`.

## Status

Work in progress (October 2026).
