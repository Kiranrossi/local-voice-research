"""Tamlish zero-shot test on a Mac (plan B when no cloud GPU is available).

    python src/run_mac_tamlish.py --prepare          # builds ~/lvr-data/own_ta and ~/lvr-data/indicvoices_ta
    python src/run_mac_tamlish.py whisper-turbo-ta   # one model per command
    python src/run_mac_tamlish.py --score            # scores everything in ~/lvr-results-tamlish
    python src/run_mac_tamlish.py --list
"""
import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import tamlish as TL  # noqa: E402
from run_mac import DATA, THREADS, _mlx_peak_gb, mac_name  # noqa: E402

RESULTS = os.path.expanduser("~/lvr-results-tamlish")
SETS = ("own_ta", "indicvoices_ta")


class WhisperMLX:
    def __init__(self, repo, language):
        self.repo, self.language = repo, language

    def load(self):
        import mlx_whisper
        self.t = mlx_whisper.transcribe
        self.t(np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.repo, language="ta")  # loads and caches

    def transcribe(self, clip):
        return self.t(clip["audio"], path_or_hf_repo=self.repo, language=self.language, temperature=0.0,
                      task="transcribe")["text"]

    def peak_mem_gb(self):
        return _mlx_peak_gb()


class HFWhisperMPS:
    """A transformers Whisper checkpoint on the Apple GPU through PyTorch MPS."""

    def __init__(self, model_id, language="ta"):
        self.model_id, self.language = model_id, language

    def load(self):
        if hasattr(self, "pipe"):
            return
        import torch
        from transformers import pipeline
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline("automatic-speech-recognition", model=self.model_id, device=dev,
                             dtype=torch.float16 if dev == "mps" else torch.float32)

    def transcribe(self, clip):
        return self.pipe({"raw": clip["audio"].copy(), "sampling_rate": 16000},
                         generate_kwargs={"language": self.language, "task": "transcribe"})["text"]


class IndicConformerCPU:
    def load(self):
        if hasattr(self, "m"):
            return
        import torch
        from transformers import AutoModel
        torch.set_num_threads(THREADS)
        self.m = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    def transcribe(self, clip):
        import torch
        out = self.m(torch.from_numpy(clip["audio"].copy()).unsqueeze(0), "ta", "rnnt")
        return out if isinstance(out, str) else (out[0] if isinstance(out, (list, tuple)) else str(out))


MODELS = {
    # name: (factory, model name, variant, profile, runtime)
    "whisper-turbo-ta": (lambda: WhisperMLX("mlx-community/whisper-large-v3-turbo", "ta"),
                         "whisper-large-v3-turbo", "language=ta", "apple_silicon", "mlx-whisper"),
    "whisper-turbo-auto": (lambda: WhisperMLX("mlx-community/whisper-large-v3-turbo", None),
                           "whisper-large-v3-turbo", "language=auto", "apple_silicon", "mlx-whisper"),
    "whisper-large-v3-ta": (lambda: WhisperMLX("mlx-community/whisper-large-v3-mlx", "ta"),
                            "whisper-large-v3", "language=ta", "apple_silicon", "mlx-whisper"),
    "whisper-tamil-medium": (lambda: HFWhisperMPS("vasista22/whisper-tamil-medium"),
                             "whisper-tamil-medium", "vasista22, language=ta", "apple_silicon", "transformers (MPS)"),
    "indicconformer": (lambda: IndicConformerCPU(), "indicconformer-600m", "ta, rnnt", "cpu", "onnx (custom code)"),
}


def prepare():
    import soundfile as sf

    src = os.path.join(DATA, "recordings", "manifest.jsonl")
    H.write_manifest([c for c in H.read_manifest(src) if c["id"].startswith("ta_")], os.path.join(DATA, "own_ta", "manifest.jsonl"))
    print("own_ta:", len(H.read_manifest(os.path.join(DATA, "own_ta", "manifest.jsonl"))), "clips")

    out = os.path.join(DATA, "indicvoices_ta")
    if os.path.exists(os.path.join(out, "manifest.jsonl")):
        print("indicvoices_ta: already prepared"); return
    from datasets import Audio, load_dataset
    os.makedirs(out, exist_ok=True)
    ds = load_dataset("ai4bharat/IndicVoices", "tamil", split="valid", streaming=True)
    cols = list(ds.features.keys()) if ds.features else ["audio_filepath", "text"]
    audio_col = next(k for k in ("audio_filepath", "audio") if k in cols)
    text_col = next(k for k in ("text", "transcript", "verbatim", "normalized") if k in cols)
    print("columns:", cols)
    ds = ds.cast_column(audio_col, Audio(decode=False)).shuffle(seed=42, buffer_size=300)
    rows = []
    for ex in ds:
        a = ex[audio_col]
        raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        dur = len(audio) / 16000
        if not (2.0 <= dur <= 20.0) or not ex[text_col]:
            continue
        cid = f"iv_ta_{len(rows):03d}"
        path = os.path.join(out, f"{cid}.wav")
        sf.write(path, audio, 16000, subtype="PCM_16")
        meta = {k: ex[k] for k in ("gender", "age_group", "district", "state", "task_name", "scenario") if isinstance(ex.get(k), str)}
        rows.append({"id": cid, "audio_filepath": path, "text": ex[text_col], "duration": round(dur, 3), **meta})
        print(f"  {cid}: {dur:.1f}s")
        if len(rows) >= 30:
            break
    H.write_manifest(rows, os.path.join(out, "manifest.jsonl"))
    print("indicvoices_ta:", len(rows), "clips; example:", rows[0]["text"][:80])


def score():
    import pandas as pd
    rows = TL.score_results(RESULTS)
    cols = ["model", "variant", "test_set", "strict_wer", "mer", "phon_cer", "en_recall", "wrong_language",
            "phon_cer_light", "phon_cer_medium", "phon_cer_heavy", "latency_median_s"]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    if a.score:
        return score()
    if a.list or a.model not in MODELS:
        print("Models:\n  " + "\n  ".join(MODELS)); return
    factory, name, variant, profile, runtime = MODELS[a.model]
    model = factory()
    device = mac_name() + (" CPU" if profile == "cpu" else " GPU")
    for ts in SETS:
        m = os.path.join(DATA, ts, "manifest.jsonl")
        if not os.path.exists(m):
            print(f"skip {ts}: run --prepare first"); continue
        H.evaluate(model, name, m, ts, RESULTS, variant=variant, runtime=runtime, quantization="fp16",
                   normaliser=TL.normalise, objective="obj2", notes="tamlish strict normaliser",
                   hardware_profile=profile, device_name=device)


if __name__ == "__main__":
    main()
