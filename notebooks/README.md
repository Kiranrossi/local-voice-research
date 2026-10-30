# Notebooks

All three run on a free Google Colab T4 GPU. Open one with the badge links in the main README,
run the cells top to bottom, and each run appends a row to `results.csv` on your Google Drive.

| Notebook | What it does |
|---|---|
| `01_stt_colab.ipynb` | Objective 1, speech-to-text: 7 models on LibriSpeech, Svarah (Indian-accented English), own recordings and no-speech clips |
| `02_tts_colab.ipynb` | Objective 1, text-to-speech: 4 models x 20 sentences; speed, memory, UTMOS and round-trip WER |
| `03_tamlish_colab.ipynb` | Objective 2: zero-shot Tamlish and Tamil test of 5 setups, scored with the code-mixed metrics in `src/tamlish.py` |

Gated models and datasets need a Hugging Face token. In Colab, store it in the Secrets panel (key icon) as `HF_TOKEN`. Never paste it into a code cell.
