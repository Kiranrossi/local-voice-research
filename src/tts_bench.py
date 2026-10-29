"""TTS benchmark helpers for local-voice-research.

A TTS model is an object with:
    load()                -> loads weights (timed separately)
    synthesize(text)      -> (audio float32 numpy array, sample_rate)

bench() generates the same 20 sentences with every model, saves each WAV at the model's
native sample rate, and appends one row to results.csv (task="tts").
score() runs afterwards, in a clean session: round-trip intelligibility (an STT model
transcribes the generated audio, WER against the input text) and predicted MOS (UTMOS).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

import harness as H

# 10 everyday assistant replies + the 10 English stress sentences used for STT
SENTENCES = [
    ("g01", "Your order has been placed and will arrive by Friday evening."),
    ("g02", "I couldn't find any meetings on your calendar for tomorrow."),
    ("g03", "The weather today is mostly sunny, with a light breeze in the afternoon."),
    ("g04", "Sure, I've set an alarm for six thirty tomorrow morning."),
    ("g05", "Your train is running about fifteen minutes late."),
    ("g06", "I've sent the message to your manager."),
    ("g07", "Would you like me to read your latest emails?"),
    ("g08", "The nearest pharmacy is two hundred metres away and is open until ten."),
    ("g09", "Your electricity bill for this month has been paid successfully."),
    ("g10", "Sorry, I didn't catch that. Could you please repeat it?"),
    ("en_01", "Transfer ₹2,499 to Karthikeyan on 15th August."),
    ("en_02", "Book a train from Chennai Egmore to Tiruchirappalli for 3rd December."),
    ("en_03", "Call 98401 23456 and ask for Mr. Subramanian."),
    ("en_04", "My complaint reference number is TN-4471-B, check the status."),
    ("en_05", "Set a reminder for 7:45 PM to pay the LIC premium of ₹12,350."),
    ("en_06", "How far is Thiruvananthapuram from Kanyakumari by road?"),
    ("en_07", "Schedule a meeting with Priya Venkataraman at 10:30 AM on 2nd November."),
    ("en_08", "Send ₹1.5 lakh to the HDFC account ending 4421."),
    ("en_09", "What is the weather in Coimbatore for the next 3 days?"),
    ("en_10", "Order 2 kg of tomatoes and 500 grams of paneer from BigBasket."),
]

TTS_SET = "tts_en20"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def bench(model, model_name: str, tts_root: str, results_dir: str, variant: str = "",
          quantization: str = "fp16", runtime: str = "", notes: str = "", warmup: int = 1,
          hardware_profile: str | None = None, device_name: str | None = None) -> dict:
    import soundfile as sf

    out = Path(tts_root) / f"{_slug(model_name)}__{hardware_profile or H.device_info()[0]}"
    out.mkdir(parents=True, exist_ok=True)

    H._reset_peak_mem()
    t0 = time.perf_counter()
    model.load()
    H._sync()
    load_time = time.perf_counter() - t0

    for _, text in SENTENCES[:warmup]:
        try:
            model.synthesize(text)
        except Exception as e:
            print(f"  warm-up failed: {type(e).__name__}: {e}"[:200])

    rows, gen_times, total_audio, total_gen, failed = [], [], 0.0, 0.0, 0
    for sid, text in SENTENCES:
        H._sync()
        t = time.perf_counter()
        try:
            audio, sr = model.synthesize(text)
            err = ""
        except Exception as e:
            audio, sr, err = np.zeros(0, dtype=np.float32), 16000, f"{type(e).__name__}: {e}"[:300]
        H._sync()
        dt = time.perf_counter() - t
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        dur = len(audio) / sr if sr else 0.0
        path = out / f"{sid}.wav"
        if len(audio):
            peak = np.abs(audio).max()
            if peak > 1.0:
                audio = audio / peak
            sf.write(path, audio, int(sr), subtype="PCM_16")
            gen_times.append(dt)
            total_audio += dur
            total_gen += dt
        else:
            failed += 1
        rows.append({"id": sid, "text": text, "audio_filepath": str(path) if len(audio) else "",
                     "duration": round(dur, 3), "gen_s": round(dt, 4), "sample_rate": int(sr), "error": err})
        print(f"  {sid}: {dur:.1f}s audio in {dt:.2f}s" + (f"  FAILED {err}" if err else ""))

    peak_mem = model.peak_mem_gb() if hasattr(model, "peak_mem_gb") else H._peak_mem_gb()
    H.write_manifest(rows, str(out / "manifest.jsonl"))

    profile, device = H.device_info()
    profile, device = hardware_profile or profile, device_name or device
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    row = {
        "run_id": f"{stamp}_{_slug(model_name)}_{TTS_SET}", "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "objective": "obj1", "task": "tts", "model": model_name, "variant": variant, "quantization": quantization,
        "runtime": runtime, "hardware_profile": profile, "device_name": device, "test_set": TTS_SET,
        "n_clips": len(rows), "rtf": round(total_gen / total_audio, 4) if total_audio else "",
        "latency_median_s": round(statistics.median(gen_times), 4) if gen_times else "",
        "latency_p90_s": round(sorted(gen_times)[int(0.9 * (len(gen_times) - 1))], 4) if gen_times else "",
        "peak_mem_gb": round(peak_mem, 2), "load_time_s": round(load_time, 1),
        "notes": "; ".join(x for x in [notes, f"audio_dir={out.name}", f"{failed} sentences failed" if failed else ""] if x),
    }
    csv_path = Path(results_dir) / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=H.RESULTS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{model_name}: RTF {row['rtf']} | median {row['latency_median_s']} s per sentence | "
          f"peak {row['peak_mem_gb']} GB | load {row['load_time_s']} s" + (f" | {failed} FAILED" if failed else ""))
    return row


def score(tts_root: str, results_dir: str, stt_transcribe, mos_predict=None,
          out_name: str = "tts_scored.csv") -> list[dict]:
    """Scores every generated TTS folder.

    stt_transcribe(audio_16k) -> text      (used for round-trip intelligibility)
    mos_predict(audio_16k) -> float        (predicted naturalness, 1 to 5), optional
    """
    out_rows = []
    for d in sorted(Path(tts_root).iterdir()):
        m = d / "manifest.jsonl"
        if not m.exists():
            continue
        items = [json.loads(l) for l in m.read_text(encoding="utf-8").splitlines() if l.strip()]
        errs, lens, ent_errs, ent_lens, moses, per = [], [], [], [], [], []
        for it in items:
            if not it.get("audio_filepath"):
                continue
            p = d / Path(it["audio_filepath"]).name
            audio = H.load_audio(str(p))
            hyp = stt_transcribe(audio)
            ref_n, hyp_n = H.normalise_en(it["text"]), H.normalise_en(hyp)
            we, wl, _, _ = H.utterance_errors(ref_n, hyp_n)
            errs.append(we); lens.append(wl)
            if it["id"].startswith("en_"):
                ent_errs.append(we); ent_lens.append(wl)
            mos = float(mos_predict(audio)) if mos_predict else float("nan")
            moses.append(mos)
            per.append({**it, "stt_hyp": hyp, "ref_norm": ref_n, "hyp_norm": hyp_n, "word_errors": we, "utmos": round(mos, 3)})
        H.write_manifest(per, str(d / "scored.jsonl"))
        lo, hi = H.bootstrap_ci(errs, lens)
        row = {"audio_dir": d.name, "n": len(per),
               "roundtrip_wer": round(H.corpus_rate(errs, lens), 4), "wer_ci_low": round(lo, 4), "wer_ci_high": round(hi, 4),
               "roundtrip_wer_everyday": round(H.corpus_rate([e for e, p in zip(errs, per) if p["id"].startswith("g")],
                                                             [l for l, p in zip(lens, per) if p["id"].startswith("g")]), 4),
               "roundtrip_wer_stress": round(H.corpus_rate(ent_errs, ent_lens), 4),
               "utmos_mean": round(float(np.nanmean(moses)), 3) if moses else ""}
        out_rows.append(row)
        print(f"{d.name}: round-trip WER {row['roundtrip_wer']:.1%} (everyday {row['roundtrip_wer_everyday']:.1%}, "
              f"stress {row['roundtrip_wer_stress']:.1%}) | UTMOS {row['utmos_mean']}")
    with open(Path(results_dir) / out_name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["audio_dir"])
        w.writeheader()
        w.writerows(out_rows)
    return out_rows
