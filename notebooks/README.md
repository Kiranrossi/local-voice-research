# Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `00_harness.ipynb` | Shared test harness: load clips, normalise text, compute WER/CER/RTF/memory, append to `results/results.csv` | To do (Day 1) |
| `01_stt_english.ipynb` | Objective 1: STT models on LibriSpeech, Svarah, own recordings | To do (Day 1) |
| `02_tts_english.ipynb` | Objective 1: TTS models, speed, memory, predicted MOS, round-trip WER | To do (Day 1) |
| `03_tamlish_zero_shot.ipynb` | Objective 2: zero-shot Tamlish STT test | To do (Day 2) |

Gated models and datasets need a Hugging Face token. In Colab, store it in the Secrets panel (key icon) as `HF_TOKEN`. Never paste it into a code cell.
