# Results

Each environment writes its own folder. Every model run appends one row to `results.csv`
and saves per-clip outputs (reference, model output, latency) to `outputs/<run_id>.jsonl`.

| Folder | Hardware | Contents |
|--------|----------|----------|
| `colab_t4/` | Google Colab, NVIDIA Tesla T4 | English STT (LibriSpeech, Svarah, own recordings, no-speech clips); English TTS (4 models x 20 sentences): `tts_scored.csv` and `tts/<model>/scored.jsonl` |
| `colab_t4_tamlish/` | Google Colab, NVIDIA Tesla T4 | Objective 2: zero-shot Tamlish (30 own recordings) and Tamil (30 IndicVoices validation clips); `tamlish_scored.csv` has strict WER, MER, phonetic CER, English-word recall, wrong-language rate |
| `mac_m2/` | MacBook Air M2, 8 GB (CPU-only, Apple GPU via MLX, CoreML) | English STT on 50 Svarah clips, own recordings, no-speech clips; Kokoro TTS on CPU |

- `results.csv`: raw results, scored at run time
- `results_scored.csv`: all runs re-scored with the final normaliser (`whisper_english+indian_v2`), latest run per model and test set only. **Use this file for the report.**

Scoring v2 adds Indian-English formatting rules on top of Whisper's English normaliser
(₹/Rs/rupees, digit grouping such as 7:45 vs 745, spelled letters such as T N vs TN),
applied identically to references and every model. See `src/harness.py`.

Svarah transcripts are from AI4Bharat's Svarah dataset (CC-BY-4.0).

Note: the first Parakeet ONNX run on the Mac silently used CoreML for part of the model (ONNX Runtime's default on macOS). It is kept, relabelled `apple_coreml`; the true CPU-only run forces `CPUExecutionProvider`.

TTS note: the Kokoro run in `colab_t4` used kokoro-onnx on Colab's 2-vCPU CPU (its folder name says nvidia_gpu because the GPU was present but unused). The fair CPU measurement is in `mac_m2`. The first Qwen3-TTS row (20 failed) is the fp16 attempt that crashed on the T4; the fp32 rerun is the valid one.

Tamlish note: IndicConformer's peak memory reads ~0 GB because it runs through ONNX Runtime, which PyTorch's memory counter cannot see; treat it as not measured. The first whisper-tamil-medium runs (all clips failed: outdated generation config) are superseded by the rerun.
