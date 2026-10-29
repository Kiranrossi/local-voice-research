"""Shared test harness for local-voice-research.

Every model is run through the same code: same clips, same text normaliser,
same timing and memory measurement, and every run appends one row to
results.csv plus one JSONL file of per-clip outputs.

A model is a small object with two methods:
    load()                 -> loads weights (not timed)
    transcribe(clip)       -> str, where clip is a dict with "path" and "audio"
                              (float32 numpy array, 16 kHz mono)
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import random
import re
import statistics
import time
import unicodedata
from pathlib import Path

import numpy as np

SR = 16000

RESULTS_COLUMNS = [
    "run_id", "timestamp", "objective", "task", "model", "variant", "quantization",
    "runtime", "hardware_profile", "device_name", "test_set", "n_clips",
    "wer", "wer_ci_low", "wer_ci_high", "cer", "hallucination_rate",
    "rtf", "latency_median_s", "latency_p90_s", "peak_mem_gb", "load_time_s", "notes",
]


# ---------------------------------------------------------------- text normalisation

_basic_punct = re.compile(r"[^\w\s']", flags=re.UNICODE)
_brackets = re.compile(r"[<\[][^>\]]*[>\]]|\([^)]*\)")


def _basic_normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _brackets.sub(" ", text)
    text = _basic_punct.sub(" ", text)
    return " ".join(text.split())


try:  # Whisper's English normaliser (same family the Open ASR Leaderboard uses)
    from whisper_normalizer.english import EnglishTextNormalizer as _ETN

    _english = _ETN()
    _BASE_NAME = "whisper_english"

    def _base(text: str) -> str:
        return " ".join(_english(text or "").split())
except Exception:  # pragma: no cover - fallback when the package is missing
    _BASE_NAME = "basic"

    def _base(text: str) -> str:
        return _basic_normalise(text or "")


# Indian-English formatting rules (scoring v2). They remove differences in *how* a
# correct answer is written, applied identically to references and every model:
#   ₹ / Rs / Rs. / INR  -> "rupees"          (currency word vs symbol)
#   7:45, 7.45, 7 45     -> 745              (digit grouping and separators)
#   t n, p m             -> tn, pm           (spelled-out letters)
_currency_pre = re.compile(r"₹\s*")
_currency_post = re.compile(r"\b(?:rs|inr)\b")
_digit_join = re.compile(r"(?<=\d)[\s.:,](?=\d)")
_letters = re.compile(r"\b([a-z])(?: ([a-z]))+\b")
NORMALISER_NAME = f"{_BASE_NAME}+indian_v2"


def _join_letters(m: re.Match) -> str:
    return m.group(0).replace(" ", "")


def normalise_en(text: str) -> str:
    text = _currency_pre.sub(" rupees ", text or "")
    text = _base(text)
    text = _currency_post.sub("rupees", text)
    text = re.sub(r"\brupees(?: rupees)+\b", "rupees", text)
    prev = None
    while prev != text:  # repeat so 98401 23 456 fully joins
        prev, text = text, _digit_join.sub("", text)
    text = _letters.sub(_join_letters, text)
    return " ".join(text.split())


# ---------------------------------------------------------------- error rates

def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token lists (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def utterance_errors(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """Returns (word_errors, ref_words, char_errors, ref_chars) for already-normalised text."""
    rw, hw = ref.split(), hyp.split()
    rc, hc = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return _edit_distance(rw, hw), len(rw), _edit_distance(rc, hc), len(rc)


def corpus_rate(errors: list[int], lengths: list[int]) -> float:
    total = sum(lengths)
    return sum(errors) / total if total else float("nan")


def bootstrap_ci(errors: list[int], lengths: list[int], n_boot: int = 1000, seed: int = 0):
    """95% bootstrap confidence interval for corpus-level WER, resampling utterances."""
    if not errors:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(errors)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(corpus_rate([errors[i] for i in idx], [lengths[i] for i in idx]))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


# ---------------------------------------------------------------- data

def load_audio(path: str) -> np.ndarray:
    """Load any audio file as float32 mono at 16 kHz."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        except ImportError:  # light fallback when librosa is not installed
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), SR)
            audio = resample_poly(audio, SR // g, int(sr) // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


def write_manifest(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: str) -> list[dict]:
    base = Path(path).parent
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                p = Path(r["audio_filepath"])
                p = p if p.is_absolute() else base / p
                if not p.exists() and (base / p.name).exists():  # manifest moved to another machine
                    p = base / p.name
                r["audio_filepath"] = str(p)
                rows.append(r)
    return rows


def make_noise_set(out_dir: str, seed: int = 0) -> str:
    """10 clips with no speech: 5 near-silent, 5 noise. Used to measure hallucination."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(10):
        dur = 3.0 + i * 0.5
        n = int(dur * SR)
        if i < 5:
            audio = rng.normal(0, 0.0005 * (i + 1), n)  # very quiet room tone
            kind = "silence"
        else:
            white = rng.normal(0, 1, n)
            # crude pink-ish noise: smooth white noise, then scale
            k = 2 ** (i - 3)
            audio = np.convolve(white, np.ones(k) / k, mode="same")
            audio = 0.05 * audio / (np.abs(audio).max() + 1e-9)
            kind = "noise"
        p = out / f"noise_{i:02d}.wav"
        sf.write(p, audio.astype(np.float32), SR, subtype="PCM_16")
        rows.append({"id": f"noise_{i:02d}", "audio_filepath": str(p.resolve()), "text": "", "duration": dur, "kind": kind})
    manifest = str(out / "manifest.jsonl")
    write_manifest(rows, manifest)
    return manifest


# ---------------------------------------------------------------- hardware info

def device_info() -> tuple[str, str]:
    """Returns (hardware_profile, device_name)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "nvidia_gpu", torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "apple_silicon", "Apple MPS"
    except Exception:
        pass
    import platform

    return "cpu", platform.processor() or platform.machine()


def _reset_peak_mem():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_mem_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, KB on Linux
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- the runner

def evaluate(model, model_name: str, manifest: str, test_set: str, results_dir: str,
             variant: str = "", quantization: str = "fp16", runtime: str = "transformers",
             objective: str = "obj1", task: str = "stt", notes: str = "",
             normaliser=normalise_en, max_clips: int | None = None, warmup: int = 2,
             hardware_profile: str | None = None, device_name: str | None = None) -> dict:
    """Runs one model on one test set, appends a row to results.csv, saves per-clip outputs."""
    clips = read_manifest(manifest)
    if max_clips:
        clips = clips[:max_clips]
    for c in clips:
        c["audio"] = load_audio(c["audio_filepath"])
        c["path"] = c["audio_filepath"]
        c["duration"] = len(c["audio"]) / SR

    _reset_peak_mem()
    t0 = time.perf_counter()
    model.load()
    _sync()
    load_time = time.perf_counter() - t0

    for c in clips[:warmup]:  # warm-up runs are not timed or scored
        try:
            model.transcribe(c)
        except Exception as e:
            print(f"  warm-up failed on {c.get('id')}: {type(e).__name__}: {e}"[:200])

    outputs, latencies = [], []
    total_audio, total_proc = 0.0, 0.0
    for i, c in enumerate(clips):
        _sync()
        t = time.perf_counter()
        try:
            hyp = model.transcribe(c) or ""
            err = ""
        except Exception as e:  # one bad clip should not kill the run
            hyp, err = "", f"{type(e).__name__}: {e}"[:300]
        _sync()
        dt = time.perf_counter() - t
        latencies.append(dt)
        total_audio += c["duration"]
        total_proc += dt
        outputs.append({"id": c.get("id", str(i)), "ref": c.get("text", ""), "hyp": hyp,
                        "duration": round(c["duration"], 3), "latency_s": round(dt, 4), "error": err,
                        **{k: v for k, v in c.items() if k not in ("audio", "path", "audio_filepath", "text", "duration", "id")}})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(clips)} clips")

    peak = model.peak_mem_gb() if hasattr(model, "peak_mem_gb") else _peak_mem_gb()

    w_err, w_len, c_err, c_len, halluc = [], [], [], [], 0
    is_noise = all(not o["ref"].strip() for o in outputs)
    for o in outputs:
        ref, hyp = normaliser(o["ref"]), normaliser(o["hyp"])
        o["ref_norm"], o["hyp_norm"] = ref, hyp
        if is_noise:
            halluc += bool(hyp)
            continue
        if not ref:
            continue
        we, wl, ce, cl = utterance_errors(ref, hyp)
        o["word_errors"] = we
        w_err.append(we); w_len.append(wl); c_err.append(ce); c_len.append(cl)

    wer = corpus_rate(w_err, w_len)
    lo, hi = bootstrap_ci(w_err, w_len)
    cer = corpus_rate(c_err, c_len)
    failed = sum(bool(o["error"]) for o in outputs)

    profile, device = device_info()
    profile, device = hardware_profile or profile, device_name or device
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{re.sub(r'[^A-Za-z0-9]+', '-', model_name)}_{test_set}"
    row = {
        "run_id": run_id, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "objective": objective, "task": task, "model": model_name, "variant": variant,
        "quantization": quantization, "runtime": runtime, "hardware_profile": profile,
        "device_name": device, "test_set": test_set, "n_clips": len(outputs),
        "wer": "" if is_noise else round(wer, 4),
        "wer_ci_low": "" if is_noise else round(lo, 4),
        "wer_ci_high": "" if is_noise else round(hi, 4),
        "cer": "" if is_noise else round(cer, 4),
        "hallucination_rate": round(halluc / len(outputs), 3) if is_noise else "",
        "rtf": round(total_proc / total_audio, 4) if total_audio else "",
        "latency_median_s": round(statistics.median(latencies), 4) if latencies else "",
        "latency_p90_s": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 4) if latencies else "",
        "peak_mem_gb": round(peak, 2), "load_time_s": round(load_time, 1),
        "notes": "; ".join(x for x in [notes, f"normaliser={NORMALISER_NAME}", "batch=1",
                                       f"{failed} clips failed" if failed else ""] if x),
    }

    out = Path(results_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    write_manifest(outputs, str(out / "outputs" / f"{run_id}.jsonl"))

    if is_noise:
        print(f"{model_name} | {test_set}: hallucinated on {halluc}/{len(outputs)} no-speech clips | "
              f"RTF {row['rtf']} | peak {row['peak_mem_gb']} GB")
    else:
        print(f"{model_name} | {test_set}: WER {wer:.2%} (95% CI {lo:.2%}-{hi:.2%}) | "
              f"RTF {row['rtf']} | median latency {row['latency_median_s']} s | peak {row['peak_mem_gb']} GB"
              + (f" | {failed} clips FAILED" if failed else ""))
    return row


def worst_examples(results_dir: str, run_id: str, n: int = 5) -> None:
    """Prints the clips with the most word errors, for error analysis."""
    path = Path(results_dir) / "outputs" / f"{run_id}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "word_errors" in r]
    for r in sorted(rows, key=lambda r: -r["word_errors"])[:n]:
        print(f"[{r['id']}] errors={r['word_errors']}\n  REF: {r['ref_norm']}\n  HYP: {r['hyp_norm']}\n")


# ---------------------------------------------------------------- re-scoring saved outputs

def rescore_all(results_dir: str, normaliser=None, out_name: str = "results_scored.csv") -> "list[dict]":
    """Re-computes accuracy for every saved run with the current normaliser.

    Timing and memory columns are copied from results.csv; only the accuracy columns change.
    Writes results_scored.csv next to results.csv and returns its rows. Only the latest run of
    each (model, test_set, hardware_profile) is kept, so crashed or superseded runs drop out.
    """
    normaliser = normaliser or normalise_en
    out = Path(results_dir)
    with open(out / "results.csv", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    latest = {}
    for r in raw:  # later rows win
        latest[(r["model"], r["test_set"], r["hardware_profile"], r.get("task", ""))] = r
    rows = []
    for r in latest.values():
        path = out / "outputs" / f"{r['run_id']}.jsonl"
        if not path.exists():
            continue
        outs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        is_noise = all(not o["ref"].strip() for o in outs)
        w_err, w_len, c_err, c_len, halluc = [], [], [], [], 0
        for o in outs:
            ref, hyp = normaliser(o["ref"]), normaliser(o["hyp"])
            if is_noise:
                halluc += bool(hyp)
                continue
            if not ref:
                continue
            we, wl, ce, cl = utterance_errors(ref, hyp)
            w_err.append(we); w_len.append(wl); c_err.append(ce); c_len.append(cl)
        r = dict(r)
        if is_noise:
            r["hallucination_rate"] = round(halluc / len(outs), 3)
        else:
            lo, hi = bootstrap_ci(w_err, w_len)
            r.update(wer=round(corpus_rate(w_err, w_len), 4), wer_ci_low=round(lo, 4),
                     wer_ci_high=round(hi, 4), cer=round(corpus_rate(c_err, c_len), 4))
        r["notes"] = re.sub(r"normaliser=[^;]*", f"normaliser={NORMALISER_NAME}", r.get("notes", ""))
        rows.append(r)
    rows.sort(key=lambda r: (r["test_set"], r["model"]))
    with open(out / out_name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Re-scored {len(rows)} runs with {NORMALISER_NAME} -> {out / out_name}")
    return rows
