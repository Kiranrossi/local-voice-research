"""Runs TTS models locally on a Mac CPU (the CPU-only profile).

Usage:
    python src/run_mac_tts.py kokoro-cpu

Writes audio to ~/lvr-tts/ and a results row to ~/lvr-results/results.csv.
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_bench as T  # noqa: E402
from run_mac import RESULTS, THREADS, mac_name  # noqa: E402

TTS_DIR = os.path.expanduser("~/lvr-tts")
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


class KokoroCPU:
    """Kokoro-82M via kokoro-onnx, forced onto the CPU provider (no CoreML)."""

    def load(self):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        cache = os.path.expanduser("~/.cache/kokoro-onnx")
        os.makedirs(cache, exist_ok=True)
        for name, url in KOKORO_FILES.items():
            path = os.path.join(cache, name)
            if not os.path.exists(path):
                print("downloading", name)
                urllib.request.urlretrieve(url, path)
        model_path, voices_path = os.path.join(cache, "kokoro-v1.0.onnx"), os.path.join(cache, "voices-v1.0.bin")
        if hasattr(Kokoro, "from_session"):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = THREADS
            sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self.k = Kokoro.from_session(sess, voices_path)
        else:  # older kokoro-onnx: choose the provider through its environment variable
            os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
            self.k = Kokoro(model_path, voices_path)

    def synthesize(self, text):
        audio, sr = self.k.create(text, voice="af_heart", speed=1.0, lang="en-us")
        return audio, sr


MODELS = {"kokoro-cpu": (KokoroCPU, "kokoro-82m", "fp32", "kokoro-onnx (CPU provider)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    a = ap.parse_args()
    factory, name, quant, runtime = MODELS[a.model]
    T.bench(factory(), name, TTS_DIR, RESULTS, variant="82M af_heart", quantization=quant, runtime=runtime,
            hardware_profile="cpu", device_name=mac_name() + " CPU", notes=f"{THREADS} threads")


if __name__ == "__main__":
    main()
