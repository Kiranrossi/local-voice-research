"""Builds the paper figures from paper/data.json (vector PDF, print sizes for IEEE two-column)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
D = json.load(open(HERE / "data.json", encoding="utf-8"))
OUT = HERE / "figures"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C = {"nvidia_gpu": "#2a78d6", "apple_silicon": "#eb6834", "cpu": "#1baf7a", "apple_coreml": "#eda100"}
MK = {"nvidia_gpu": "o", "apple_silicon": "s", "cpu": "D", "apple_coreml": "^"}
PLAT = {"nvidia_gpu": "NVIDIA T4 (Colab)", "apple_silicon": "Apple M2 GPU (MLX)", "cpu": "M2 CPU only (4 threads)",
        "apple_coreml": "M2 CoreML + CPU"}
SHORT = {"whisper-large-v3-turbo": "Whisper turbo", "whisper-large-v3-turbo+vad": "Whisper turbo+VAD",
         "parakeet-tdt-0.6b-v2": "Parakeet v2", "parakeet-tdt-0.6b-v3": "Parakeet v3", "qwen3-asr-1.7B": "Qwen3-ASR 1.7B",
         "qwen3-asr-0.6B": "Qwen3-ASR 0.6B", "moonshine-streaming-medium": "Moonshine M", "moonshine-streaming-tiny": "Moonshine T",
         "whisper-large-v3": "Whisper large-v3", "whisper-tamil-medium": "Whisper-medium (Tamil FT)",
         "indicconformer-600m": "IndicConformer 600M"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": SURF})
pct = FuncFormatter(lambda v, _: f"{v:.0f}%")


def style(ax):
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- Fig 1: STT accuracy vs latency (Svarah, Indian-accented English)
fig, ax = plt.subplots(figsize=(3.45, 2.55))
pts = [r for r in D["stt"] if r["test_set"] == "svarah"]
# label placement: (dx, dy, ha) in points; None hides a duplicate label
offsets = {("whisper-large-v3-turbo", "nvidia_gpu"): (5, -2, "left"), ("qwen3-asr-1.7B", "nvidia_gpu"): (5, -3, "left"),
           ("qwen3-asr-0.6B", "nvidia_gpu"): (2, 7, "center"), ("parakeet-tdt-0.6b-v2", "nvidia_gpu"): None,
           ("parakeet-tdt-0.6b-v3", "nvidia_gpu"): (0, 8, "center"), ("moonshine-streaming-medium", "nvidia_gpu"): (5, 0, "left"),
           ("moonshine-streaming-tiny", "nvidia_gpu"): (5, 0, "left"), ("whisper-large-v3-turbo", "apple_silicon"): (0, -8, "center"),
           ("whisper-large-v3-turbo", "cpu"): None, ("whisper-large-v3-turbo+vad", "cpu"): (0, 9, "center"),
           ("parakeet-tdt-0.6b-v2", "cpu"): (0, -8, "center"), ("moonshine-streaming-medium", "cpu"): (5, 0, "left"),
           ("parakeet-tdt-0.6b-v2", "apple_coreml"): (5, -1, "left")}
text = {("parakeet-tdt-0.6b-v3", "nvidia_gpu"): "Parakeet v2 / v3", ("whisper-large-v3-turbo+vad", "cpu"): "Whisper turbo (± VAD)"}
for r in pts:
    x, y, p = float(r["latency_median_s"]), 100 * float(r["wer"]), r["hardware_profile"]
    ax.scatter(x, y, s=22, color=C[p], marker=MK[p], edgecolor=SURF, linewidth=0.8, zorder=3)
    o = offsets.get((r["model"], p), (5, 0, "left"))
    if o is None:
        continue
    ax.annotate(text.get((r["model"], p), SHORT[r["model"]]), (x, y), xytext=o[:2], textcoords="offset points",
                fontsize=5.6, color=INK, ha=o[2], va="center")
ax.set_xscale("log")
ax.set_xlim(0.06, 14)
ax.set_ylim(3, 25)
ax.set_yticks([5, 10, 15, 20, 25])
ax.yaxis.set_major_formatter(pct)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
ax.set_xlabel("Median latency per sentence (log scale, batch 1)")
ax.set_ylabel("WER on Svarah (Indian-accented English)")
style(ax)
hs = [plt.Line2D([], [], marker=MK[p], color=C[p], ls="", markersize=4.5, label=PLAT[p]) for p in C]
ax.legend(handles=hs, loc="upper right", fontsize=5.6, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "stt_accuracy_latency.pdf")

# ---------------- Fig 2: Tamlish zero-shot (own clips vs IndicVoices)
tl = {(r["model"], r["variant"], r["test_set"]): r for r in D["tamlish"]}
order = [("indicconformer-600m", "ta, rnnt"), ("whisper-large-v3", "language=ta"),
         ("whisper-tamil-medium", "vasista22, language=ta"), ("whisper-large-v3-turbo", "language=ta"),
         ("whisper-large-v3-turbo", "language=auto")]
labels = ["IndicConformer 600M", "Whisper large-v3", "Whisper-medium\n(Tamil FT)", "Whisper turbo\n(ta forced)",
          "Whisper turbo\n(auto-detect)"]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1.25, 1]})
import numpy as np
y = np.arange(len(order))[::-1]
h = 0.36
ax = axs[0]
sw = [100 * float(tl[(m, v, "own_ta")]["strict_wer"]) for m, v in order]
pc = [100 * float(tl[(m, v, "own_ta")]["phon_cer"]) for m, v in order]
ax.barh(y + h / 2 + 0.01, sw, height=h, color="#c3c2b7", label="Strict WER (script-sensitive)", zorder=3)
ax.barh(y - h / 2 - 0.01, pc, height=h, color="#2a78d6", label="Phonetic CER (script-agnostic)", zorder=3)
for yy, a, b in zip(y, sw, pc):
    ax.text(a + 1, yy + h / 2, f"{a:.0f}%", va="center", fontsize=5.8, color=INK2)
    ax.text(b + 1, yy - h / 2, f"{b:.1f}%", va="center", fontsize=5.8, color=INK)
ax.set_yticks(y, labels, fontsize=6)
ax.set_xlim(0, 80)
ax.xaxis.set_major_formatter(pct)
ax.set_title("(a) Own Tamlish recordings (30 clips)", fontsize=7, color=INK, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=5.8)
style(ax); ax.grid(axis="y", visible=False)
ax = axs[1]
iv = [100 * float(tl[(m, v, "indicvoices_ta")]["phon_cer"]) for m, v in order]
ax.barh(y, [min(v, 100) for v in iv], height=0.55, color="#eb6834", zorder=3)
for yy, v in zip(y, iv):
    txt = f"{v:.0f}% (runaway output)" if v > 100 else f"{v:.1f}%"
    ax.text(min(v, 100) + 1.5 if v <= 100 else 98, yy, txt, va="center", ha="left" if v <= 100 else "right",
            fontsize=5.8, color=INK if v <= 100 else SURF)
ax.set_yticks(y, [""] * len(y))
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(pct)
ax.set_title("(b) IndicVoices Tamil, phonetic CER (30 clips)", fontsize=7, color=INK, loc="left")
style(ax); ax.grid(axis="y", visible=False)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / "tamlish_zero_shot.pdf")

# ---------------- Fig 3: TTS quality vs speed
q = {r["audio_dir"].split("__")[0]: r for r in D["tts_quality"]}
slug = {"kokoro-82m": "kokoro-82m", "chatterbox": "chatterbox", "qwen3-tts-0.6b": "qwen3-tts-0-6b", "voxcpm2": "voxcpm2"}
name = {"kokoro-82m": "Kokoro-82M", "chatterbox": "Chatterbox", "qwen3-tts-0.6b": "Qwen3-TTS 0.6B", "voxcpm2": "VoxCPM2"}
fig, ax = plt.subplots(figsize=(3.45, 2.2))
for r in D["tts_speed"]:
    m = r["model"]; cpu = "CPU" in r["runtime"]
    p = ("cpu" if r["folder"] == "mac_m2" else "colab_cpu") if cpu else "nvidia_gpu"
    col = {"cpu": C["cpu"], "nvidia_gpu": C["nvidia_gpu"], "colab_cpu": "#6250d6"}[p]
    mk = {"cpu": "D", "nvidia_gpu": "o", "colab_cpu": "v"}[p]
    x, yv = float(r["rtf"]), float(q[slug[m]]["utmos_mean"])
    ax.scatter(x, yv, s=24, color=col, marker=mk, edgecolor=SURF, linewidth=0.8, zorder=3)
    lab = name[m] + (" (M2 CPU)" if p == "cpu" else " (Colab 2-vCPU)" if p == "colab_cpu" else "")
    dx, dy, hal = {"chatterbox": (-5, -6, "right"), "kokoro-82m": (5, 6, "left") if p == "cpu" else (5, 6, "left"),
                   "qwen3-tts-0.6b": (0, -8, "center"), "voxcpm2": (0, 8, "center")}[m]
    ax.annotate(lab, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, color=INK, va="center", ha=hal)
ax.axvline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.text(0.95, 3.53, "real time", fontsize=5.6, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_xlim(0.15, 4.5); ax.set_ylim(3.5, 4.7)
from matplotlib.ticker import FixedLocator, NullFormatter
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Real-time factor (generation time ÷ audio length, log)")
ax.set_ylabel("UTMOS (predicted naturalness, 1–5)")
style(ax)
hs = [plt.Line2D([], [], marker=m, color=c, ls="", markersize=4.5, label=l) for m, c, l in
      (("o", C["nvidia_gpu"], "NVIDIA T4 GPU"), ("D", C["cpu"], "M2 CPU"), ("v", "#6250d6", "Colab CPU (2 vCPU)"))]
ax.legend(handles=hs, loc="lower left", fontsize=5.6)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "tts_quality_speed.pdf")
print("figures written")
