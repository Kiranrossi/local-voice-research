# Results

Each environment writes its own folder. Every model run appends one row to `results.csv`
and saves per-clip outputs (reference, model output, latency) to `outputs/<run_id>.jsonl`.

| Folder | Hardware | Contents |
|--------|----------|----------|
| `colab_t4/` | Google Colab, NVIDIA Tesla T4 | English STT: LibriSpeech, Svarah, own recordings, no-speech clips |

- `results.csv`: raw results, scored at run time
- `results_scored.csv`: all runs re-scored with the final normaliser (`whisper_english+indian_v2`), latest run per model and test set only. **Use this file for the report.**

Scoring v2 adds Indian-English formatting rules on top of Whisper's English normaliser
(₹/Rs/rupees, digit grouping such as 7:45 vs 745, spelled letters such as T N vs TN),
applied identically to references and every model. See `src/harness.py`.

Svarah transcripts are from AI4Bharat's Svarah dataset (CC-BY-4.0).
