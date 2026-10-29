"""Runs the STT models locally on a Mac (CPU-only and Apple GPU profiles).

Usage (one model per command, so memory is measured per model):
    python src/run_mac.py faster-whisper-turbo-cpu
    python src/run_mac.py --list

Expects test data in ~/lvr-data (folders svarah/, noise/, recordings/) and writes
results to ~/lvr-results (results.csv + outputs/).
"""
import argparse
import os
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

DATA = os.path.expanduser("~/lvr-data")
RESULTS = os.path.expanduser("~/lvr-results")
THREADS = 4  # the M2 has 4 performance cores


def mac_name() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        return f"{chip} ({mem / 2**30:.0f} GB)"
    except Exception:
        return platform.machine()


# ---------------------------------------------------------------- CPU-only models

class FasterWhisperCPU:
    def __init__(self, vad=False):
        self.vad = vad

    def load(self):
        from faster_whisper import WhisperModel
        self.m = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=THREADS)

    def transcribe(self, clip):
        segs, _ = self.m.transcribe(clip["audio"], language="en", beam_size=1, vad_filter=self.vad)
        return " ".join(s.text for s in segs)


class MoonshineCPU:
    def __init__(self, size):
        self.model_id = f"moonshine-ai/moonshine-streaming-{size}"

    def load(self):
        import torch
        from transformers import pipeline
        torch.set_num_threads(THREADS)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model_id, device="cpu")

    def transcribe(self, clip):
        audio = np.pad(clip["audio"], (0, (-len(clip["audio"])) % 320))
        max_tokens = int(clip["duration"] * 6.5) + 10
        return self.pipe({"raw": audio, "sampling_rate": 16000}, generate_kwargs={"max_new_tokens": max_tokens})["text"]


class ParakeetONNXCPU:
    """ONNX Runtime picks CoreML on Macs by default; force the CPU provider for a true CPU-only test."""
    def __init__(self, providers=("CPUExecutionProvider",)):
        self.providers = list(providers)

    def load(self):
        import onnx_asr
        self.m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", quantization="int8", providers=self.providers)

    def transcribe(self, clip):
        out = self.m.recognize(clip["path"])
        return out if isinstance(out, str) else getattr(out, "text", str(out))


# ---------------------------------------------------------------- Apple GPU (MLX) models

def _mlx_peak_gb():
    import mlx.core as mx
    for fn in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn:
            return fn() / 1e9
    return float("nan")


class WhisperMLX:
    repo = "mlx-community/whisper-large-v3-turbo"

    def load(self):
        import mlx_whisper
        self.t = mlx_whisper.transcribe
        self.t(np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.repo, language="en")  # loads and caches the model

    def transcribe(self, clip):
        return self.t(clip["audio"], path_or_hf_repo=self.repo, language="en", temperature=0.0)["text"]

    def peak_mem_gb(self):
        return _mlx_peak_gb()


class ParakeetMLX:
    def load(self):
        from parakeet_mlx import from_pretrained
        self.m = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

    def transcribe(self, clip):
        return self.m.transcribe(clip["path"]).text

    def peak_mem_gb(self):
        return _mlx_peak_gb()


MODELS = {
    # name: (factory, results model name, profile, quantization, runtime)
    "faster-whisper-turbo-cpu": (lambda: FasterWhisperCPU(), "whisper-large-v3-turbo", "cpu", "int8", "faster-whisper"),
    "faster-whisper-turbo-vad-cpu": (lambda: FasterWhisperCPU(vad=True), "whisper-large-v3-turbo+vad", "cpu", "int8", "faster-whisper"),
    "moonshine-medium-cpu": (lambda: MoonshineCPU("medium"), "moonshine-streaming-medium", "cpu", "fp32", "transformers"),
    "moonshine-tiny-cpu": (lambda: MoonshineCPU("tiny"), "moonshine-streaming-tiny", "cpu", "fp32", "transformers"),
    "parakeet-onnx-cpu": (lambda: ParakeetONNXCPU(), "parakeet-tdt-0.6b-v2", "cpu", "int8", "onnx-asr (CPU provider)"),
    "parakeet-onnx-coreml": (lambda: ParakeetONNXCPU(("CoreMLExecutionProvider", "CPUExecutionProvider")),
                             "parakeet-tdt-0.6b-v2", "apple_coreml", "int8", "onnx-asr (CoreML + CPU)"),
    "whisper-mlx": (lambda: WhisperMLX(), "whisper-large-v3-turbo", "apple_silicon", "fp16", "mlx-whisper"),
    "parakeet-mlx": (lambda: ParakeetMLX(), "parakeet-tdt-0.6b-v2", "apple_silicon", "bf16", "parakeet-mlx"),
}


def prepare_own_en():
    src = os.path.join(DATA, "recordings", "manifest.jsonl")
    dst = os.path.join(DATA, "own_en", "manifest.jsonl")
    if os.path.exists(src):
        H.write_manifest([c for c in H.read_manifest(src) if c["id"].startswith("en_")], dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="one of: " + ", ".join(MODELS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--svarah-clips", type=int, default=50)
    a = ap.parse_args()
    if a.list or not a.model:
        print("Models:\n  " + "\n  ".join(MODELS)); return
    factory, name, profile, quant, runtime = MODELS[a.model]
    prepare_own_en()
    model = factory()
    device = mac_name() + (" CPU" if profile == "cpu" else " GPU (Metal)")
    for ts, n in (("svarah", a.svarah_clips), ("own_en", None), ("noise", None)):
        m = os.path.join(DATA, ts, "manifest.jsonl")
        if not os.path.exists(m):
            print(f"skip {ts}: {m} not found"); continue
        H.evaluate(model, name, m, ts, RESULTS, quantization=quant, runtime=runtime, max_clips=n,
                   hardware_profile=profile, device_name=device,
                   notes=f"{THREADS} threads" if profile == "cpu" else "")


if __name__ == "__main__":
    main()
