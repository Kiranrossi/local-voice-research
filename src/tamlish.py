"""Scoring for Tamil-English code-mixed (Tamlish) speech recognition.

Plain WER is unfair to code-mixed output: the English word "office" may be written in Latin script
or in Tamil script (ஆபீஸ்), and both are correct. We therefore report several metrics:

  strict_wer     word error rate after light normalisation (script-sensitive)
  mer            mixed error rate: English words count as one unit, Tamil words as characters
                 (the unit used in code-switching challenges such as MUCS 2021)
  phon_cer       script-agnostic: every word is romanised and reduced to a coarse phonetic key,
                 then character errors are counted on the keys (also ignores word segmentation)
  en_recall      share of English words in the reference that appear in the output, in either
                 script (fuzzy phonetic match)
  wrong_language share of outputs that came back almost entirely in Latin script (translated or
                 English-only) when the reference is mostly Tamil

The phonetic key is deliberately coarse (it drops vowels after the first letter and merges sounds
Tamil script does not distinguish, such as k/g, p/b/f, t/d, s/ch/sh). It is an approximation and
is reported as such; examples are printed so a native speaker can sanity-check it.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path

import harness as H

_TAMIL = re.compile(r"[஀-௿]")
_LATIN = re.compile(r"[A-Za-z]")
_PUNCT = set("!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~“”‘’…।–—«»¿¡")

# ---------------------------------------------------------------- normalisation and tokens


def normalise(text: str) -> str:
    """NFC, lower-case, hyphens to spaces, punctuation removed. Keeps Tamil vowel signs intact."""
    text = unicodedata.normalize("NFC", text or "").lower().replace("-", " ")
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    return " ".join(text.split())


def is_tamil(tok: str) -> bool:
    return bool(_TAMIL.search(tok))


def is_latin(tok: str) -> bool:
    return bool(_LATIN.search(tok)) and not is_tamil(tok)


# ---------------------------------------------------------------- romanisation

_V = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu", "எ": "e", "ஏ": "ee",
      "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au", "ஃ": "h"}
_C = {"க": "k", "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t", "ண": "n", "த": "th", "ந": "n",
      "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
      "ற": "r", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h"}
_S = {"ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu", "ெ": "e", "ே": "ee", "ை": "ai",
      "ொ": "o", "ோ": "oo", "ௌ": "au"}
_VIRAMA = "்"


def romanise(text: str) -> str:
    """Simple, consistent Tamil-to-Latin romanisation (Latin text passes through)."""
    out, i, t = [], 0, unicodedata.normalize("NFC", text)
    while i < len(t):
        ch = t[i]
        if ch in _C:
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if nxt == _VIRAMA:
                out.append(_C[ch]); i += 2; continue
            if nxt in _S:
                out.append(_C[ch] + _S[nxt]); i += 2; continue
            out.append(_C[ch] + "a"); i += 1; continue
        out.append(_V.get(ch, ch if ch not in _S and ch != _VIRAMA else ""))
        i += 1
    return "".join(out)


def phon_key(word: str) -> str:
    """Coarse phonetic key that treats an English word and its Tamil-script spelling alike."""
    w = romanise(word).lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return ""
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"g(?=[eiy])", "j", w)  # soft g as in message, gym
    for a, b in (("ph", "p"), ("sh", "s"), ("ch", "s"), ("zh", "l"), ("th", "t"), ("dh", "t"),
                 ("ck", "k"), ("ng", "n"), ("ny", "n"), ("x", "ks"), ("q", "k")):
        w = w.replace(a, b)
    w = w.translate(str.maketrans({"c": "k", "g": "k", "f": "p", "b": "p", "d": "t", "w": "v",
                                   "z": "s", "j": "s", "h": ""}))
    head = "a" if w[0] in "aeiouy" else w[0]
    tail = re.sub(r"[aeiouy]", "", w[1:])
    key = head + tail
    return re.sub(r"(.)\1+", r"\1", key)


# ---------------------------------------------------------------- metrics


def mixed_units(text: str) -> list:
    units = []
    for tok in normalise(text).split():
        units.extend(list(tok) if is_tamil(tok) else [tok])
    return units


def _similar(a: str, b: str) -> bool:
    """a, b: phonetic keys. Short keys must match exactly; longer keys need 75% similarity."""
    if not a or not b:
        return False
    if len(a) <= 3 or len(b) <= 3:
        return a == b
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def _en_hits(r_tok: list, h_tok: list) -> int:
    """English reference words found in the output, in either script, near the same position.

    Position window: within 25% of the sentence length (at least 2 words) of the expected place,
    so an unrelated Tamil word elsewhere that happens to share consonants cannot count as a hit.
    """
    hits, n_r, n_h = 0, max(len(r_tok), 1), max(len(h_tok), 1)
    h_keys = [phon_key(t) for t in h_tok]
    for i, t in enumerate(r_tok):
        if not is_latin(t):
            continue
        e = phon_key(t)
        centre = i / n_r * n_h
        win = max(2, 0.25 * n_h)
        for j, k in enumerate(h_keys):
            fused = is_tamil(h_tok[j]) and len(e) >= 2 and len(k) > len(e) and k.startswith(e)  # e.g. டீம்க்கு = team + க்கு
            if abs(j - centre) <= win and (_similar(e, k) or fused):
                hits += 1
                break
    return hits


def utterance_metrics(ref: str, hyp: str) -> dict:
    rn, hn = normalise(ref), normalise(hyp)
    r_tok, h_tok = rn.split(), hn.split()
    we = H._edit_distance(r_tok, h_tok)
    ru, hu = mixed_units(ref), mixed_units(hyp)
    me = H._edit_distance(ru, hu)
    rk = "".join(phon_key(t) for t in r_tok)
    hk = "".join(phon_key(t) for t in h_tok)
    pe = H._edit_distance(list(rk), list(hk))
    en_ref = [t for t in r_tok if is_latin(t)]
    en_hit = _en_hits(r_tok, h_tok)
    tamil_ref = sum(is_tamil(t) for t in r_tok) / max(len(r_tok), 1)
    tamil_hyp = sum(is_tamil(t) for t in h_tok) / max(len(h_tok), 1)
    return {"word_errors": we, "ref_words": len(r_tok), "mixed_errors": me, "ref_units": len(ru),
            "phon_errors": pe, "ref_phon": len(rk), "en_words": len(en_ref), "en_hits": en_hit,
            "wrong_language": tamil_ref >= 0.5 and tamil_hyp < 0.2 and bool(h_tok),
            "latin_share_hyp": round(1 - tamil_hyp, 3) if h_tok else 0.0}


def score_run(outputs: list) -> dict:
    tot = {k: 0 for k in ("word_errors", "ref_words", "mixed_errors", "ref_units", "phon_errors", "ref_phon",
                          "en_words", "en_hits", "wrong_language")}
    per = []
    for o in outputs:
        m = utterance_metrics(o["ref"], o["hyp"])
        per.append({**o, **m})
        for k in tot:
            tot[k] += int(m[k])
    n = max(len(outputs), 1)
    return {"n": len(outputs),
            "strict_wer": tot["word_errors"] / max(tot["ref_words"], 1),
            "mer": tot["mixed_errors"] / max(tot["ref_units"], 1),
            "phon_cer": tot["phon_errors"] / max(tot["ref_phon"], 1),
            "en_recall": tot["en_hits"] / tot["en_words"] if tot["en_words"] else float("nan"),
            "wrong_language": tot["wrong_language"] / n,
            "per_clip": per}


def score_results(results_dir: str, test_sets=("own_ta", "indicvoices_ta"), out_name="tamlish_scored.csv") -> list:
    """Scores the latest run of every model on the Tamlish test sets."""
    import csv

    rd = Path(results_dir)
    with open(rd / "results.csv", encoding="utf-8") as f:
        raw = [r for r in csv.DictReader(f) if r["test_set"] in test_sets]
    latest = {}
    for r in raw:
        latest[(r["model"], r["variant"], r["test_set"])] = r
    rows = []
    for (model, variant, ts), r in sorted(latest.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        outs = [json.loads(l) for l in (rd / "outputs" / f"{r['run_id']}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        s = score_run(outs)
        # mixing level (light / medium / heavy) for the 30 own clips
        by_level = {}
        if ts == "own_ta":
            for lvl, ids in (("light", range(1, 11)), ("medium", range(11, 21)), ("heavy", range(21, 31))):
                sub = [o for o in outs if int(o["id"].split("_")[1]) in ids]
                if sub:
                    by_level[lvl] = score_run(sub)["phon_cer"]
        row = {"model": model, "variant": variant, "test_set": ts, "n": s["n"],
               "strict_wer": round(s["strict_wer"], 4), "mer": round(s["mer"], 4), "phon_cer": round(s["phon_cer"], 4),
               "en_recall": "" if s["en_recall"] != s["en_recall"] else round(s["en_recall"], 3),
               "wrong_language": round(s["wrong_language"], 3),
               "phon_cer_light": round(by_level.get("light", float("nan")), 4) if by_level else "",
               "phon_cer_medium": round(by_level.get("medium", float("nan")), 4) if by_level else "",
               "phon_cer_heavy": round(by_level.get("heavy", float("nan")), 4) if by_level else "",
               "latency_median_s": r["latency_median_s"], "peak_mem_gb": r["peak_mem_gb"], "run_id": r["run_id"]}
        rows.append(row)
        H.write_manifest(s["per_clip"], str(rd / "outputs" / f"{r['run_id']}.tamlish.jsonl"))
    if rows:
        with open(rd / out_name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows
