"""Collects every number used in the paper into paper/data.json (reads only results/)."""
import csv, json, os, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import harness as H, tamlish as TL, analyze as A

R = ROOT / "results"
D = {}

# ---------------- STT (scoring v2), latest run per model/test set/profile
stt = []
for r in A.stt_rows():
    stt.append({k: r[k] for k in ("model", "runtime", "hardware_profile", "device_name", "test_set", "n_clips", "wer",
                                  "wer_ci_low", "wer_ci_high", "hallucination_rate", "rtf", "latency_median_s",
                                  "peak_mem_gb", "quantization", "run_id")} | {"folder": r["_folder"]})
D["stt"] = stt

# same-50 Svarah WER
ids50 = {f"svarah_{i:03d}" for i in range(50)}
D["svarah50"] = [{"model": r["model"], "profile": r["hardware_profile"], "runtime": r["runtime"],
                  "wer50": A.wer_on_ids(r["folder"], r["run_id"], ids50)} for r in stt if r["test_set"] == "svarah"]

# entity accuracy
ents = []
for r in stt:
    if r["test_set"] == "own_en":
        hit, tot, bk, kt = A.entity_scores(r["folder"], r["run_id"])
        ents.append({"model": r["model"], "profile": r["hardware_profile"], "runtime": r["runtime"], "hit": hit, "total": tot,
                     "by_kind": dict(bk), "kind_total": dict(kt)})
D["entities"] = ents

# ---------------- TTS
D["tts_quality"] = A.read_csv(R / "colab_t4" / "tts_scored.csv")
tts_speed = []
for folder in ("colab_t4", "mac_m2"):
    for r in A.read_csv(R / folder / "results.csv"):
        if r["task"] == "tts" and r["rtf"]:
            tts_speed.append({k: r[k] for k in ("model", "runtime", "hardware_profile", "device_name", "rtf",
                                                "latency_median_s", "peak_mem_gb", "load_time_s", "quantization")} | {"folder": folder})
D["tts_speed"] = tts_speed
listen = collections.defaultdict(lambda: {"mos": [], "y": 0, "p": 0, "n": 0})
codes = {"m1": "kokoro-82m", "m2": "chatterbox", "m3": "qwen3-tts-0.6b", "m4": "voxcpm2"}
n_l = 0
for line in (R / "listening" / "responses.txt").read_text(encoding="utf-8").splitlines():
    if line.startswith("VLT1"):
        n_l += 1
        for item in line.split(";", 3)[3].split(","):
            k, v = item.split("="); m = codes[k.rsplit("_", 1)[1]]
            listen[m]["mos"].append(int(v[0])); listen[m][v[1]] += 1
D["listening"] = {"n_listeners": n_l, "models": {m: {"mos": sum(x["mos"]) / len(x["mos"]), "n": len(x["mos"]),
                                                     "yes": x["y"], "mostly": x["p"], "no": x["n"]} for m, x in listen.items()}}

# ---------------- Tamlish
tl_rows = A.read_csv(R / "colab_t4_tamlish" / "tamlish_scored.csv")
def latin_en_rate(run_id):
    hit = tot = loops = 0
    for l in (R / "colab_t4_tamlish" / "outputs" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(l)
        r_tok, h_tok = TL.normalise(o["ref"]).split(), TL.normalise(o["hyp"]).split()
        if len(h_tok) > 2 * max(len(r_tok), 1) + 5:
            loops += 1
        lat = [t for t in h_tok if TL.is_latin(t)]
        for t in r_tok:
            if TL.is_latin(t):
                tot += 1
                if TL._en_hits([t], lat) if lat else 0:
                    hit += 1
    return (hit / tot if tot else None), loops
for r in tl_rows:
    r["latin_en_rate"], r["runaway_outputs"] = latin_en_rate(r["run_id"])
D["tamlish"] = tl_rows
json.dump(D, open(Path(__file__).parent / "data.json", "w"), indent=1, ensure_ascii=False, default=str)
print("ok")
