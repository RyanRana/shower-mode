"""Figures for the realistic-benchmark round, from tune/preds/*.npz and tune/results.jsonl.

Needs only numpy + matplotlib (the project venv has no matplotlib; the system python does):
    /usr/bin/python3 tune/figs.py
Writes tune/figs/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PREDS, OUT = HERE / "preds", HERE / "figs"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_XSIM, C_GBM, C_GBMR = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed per model
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "legend.frameon": False})

MODELS = [  # (label, preds prefix, color)
    ("xsim CNN (realistic-trained)", "xsim_main_s{s}", C_XSIM),
    ("GBM, clean-trained (old)", "gbm_cleantrain_s{s}", C_GBM),
    ("GBM, realistic-trained", "gbm_realistictrain_s{s}", C_GBMR),
]


def load(prefix: str, ev: str, split: str = "dev", seed: int = 0):
    if prefix.startswith("gbm"):
        kind = prefix.split("_")[1]
        f = PREDS / f"gbm_{kind}_{ev}_{split}_s{seed}.npz"
    else:
        f = PREDS / f"{prefix.format(s=seed)}_{ev}_{split}.npz"
    return dict(np.load(f)) if f.exists() else None


def curve(d):
    y, p = d["y"], d["p"]
    thr = np.unique(np.concatenate([np.quantile(p[y == 0], np.linspace(0, 1, 400)), [d["thr"]]]))
    fp = np.array([(p[y == 0] >= t).mean() for t in thr])
    tp = np.array([(p[y == 1] >= t).mean() for t in thr])
    return fp, tp


def fig_roc():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, prefix, col in MODELS:
        d = load(prefix, "realistic")
        if d is None:
            continue
        fp, tp = curve(d)
        ax.plot(np.maximum(fp, 1e-3), tp, color=col, lw=2, label=label)
        hit = d["p"] >= d["thr"]
        ax.plot(max(hit[d["y"] == 0].mean(), 1e-3), hit[d["y"] == 1].mean(), "o", ms=8, color=col,
                mec=SURF, mew=2)
    ax.axvline(0.02, color=INK2, lw=1, ls="--")
    ax.text(0.021, 0.02, "2% false singing", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false singing (share of non-singing windows called singing)")
    ax.set_ylabel("singing caught")
    ax.set_title("Realistic dev: catch vs false-singing (dots = OOF-calibrated operating point)", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "roc_realistic_dev.png", dpi=150)


def fig_cond_fp():
    conds = ["talk", "other_song", "bleed", "shifted"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.26
    for j, (label, prefix, col) in enumerate(MODELS):
        vals = []
        for s in (0, 1, 2):
            d = load(prefix, "realistic", seed=s)
            if d is None:
                continue
            names = [str(x) for x in d["conds"]]
            hit = d["p"] >= d["thr"]
            vals.append([hit[(d["y"] == 0) & (d["c"] == names.index(c))].mean() for c in conds])
        if not vals:
            continue
        v = np.array(vals)
        x = np.arange(len(conds)) + (j - 1) * (w + 0.02)
        ax.bar(x, v.mean(0) * 100, w, color=col, label=label,
               yerr=v.std(0) * 100 if len(v) > 1 else None, ecolor=INK2, capsize=3)
        for xi, vi in zip(x, v.mean(0)):
            ax.text(xi, vi * 100 + 0.3, f"{vi:.1%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(conds)), conds)
    ax.set_ylabel("false singing (%)")
    ax.set_title("Realistic dev: false singing by negative condition (mean, error bar = sd over seeds)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_condition_fp.png", dpi=150)


def fig_degradation():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for label, prefix, col in MODELS:
        pts = []
        for ev in ("clean", "realistic"):
            vs = []
            for s in (0, 1, 2):
                d = load(prefix, ev, seed=s)
                if d is not None:
                    vs.append((d["p"][d["y"] == 1] >= d["thr"]).mean())
            pts.append(np.mean(vs) if vs else np.nan)
        ax.plot([0, 1], np.array(pts) * 100, "-o", color=col, lw=2, ms=8, mec=SURF, mew=2)
        ax.text(1.04, pts[1] * 100, f"{label}  {pts[0]:.0%} -> {pts[1]:.0%}", va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["clean eval", "realistic eval"])
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("singing caught on dev (%)")
    ax.set_title("Clean vs realistic eval (each model's own OOF threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_clean_vs_realistic.png", dpi=150)


def fig_ablation():
    rows = [json.loads(line) for line in (HERE / "results.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("split") == "dev" and r.get("eval") == "realistic"]
    items = [("main (4 s, cross-sim, strong bleed)", "main"), ("no cross-similarity", "noxsim"),
             ("no bleed augmentation", "weakbleed"), ("3 s window", "w3")]
    labels, means, sds, fps, cols = [], [], [], [], []
    for lab, var in items:
        v = [r for r in rows if r.get("model") == "xsim" and r.get("variant") == var]
        if v:
            labels.append(lab)
            means.append(np.mean([r["caught"] for r in v]))
            sds.append(np.std([r["caught"] for r in v]) if len(v) > 1 else 0)
            fps.append(np.mean([r["false_singing"] for r in v]))
            cols.append(C_XSIM)
    for lab, kind, col in (("GBM clean-trained", "clean", C_GBM), ("GBM realistic-trained", "realistic", C_GBMR)):
        v = [r for r in rows if r.get("model") == "gbm" and r.get("train_data") == kind]
        if v:
            labels.append(lab)
            means.append(np.mean([r["caught"] for r in v]))
            sds.append(np.std([r["caught"] for r in v]))
            fps.append(np.mean([r["false_singing"] for r in v]))
            cols.append(col)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, np.array(means) * 100, 0.6, xerr=np.array(sds) * 100, color=cols, ecolor=INK2, capsize=3)
    for yi, m, f in zip(y, means, fps):
        ax.text(m * 100 + 1.5, yi, f"{m:.1%}  (false {f:.1%})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("singing caught, realistic dev (%) at OOF 2% threshold")
    ax.set_title("Ablations (error bar = sd over seeds where run)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png", dpi=150)


def fig_examples():
    f = PREDS / "xsim_examples.npz"
    if not f.exists():
        return
    d = np.load(f)
    conds = ["sung", "talk", "other_song", "bleed", "shifted"]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2), sharey=True)
    for j, c in enumerate(conds):
        vol = d[c]  # (H+2, T, L): learned heads, chroma cosine, spectral cosine
        for i, (img, ttl, cmap, lim) in enumerate(((vol[-2], "fixed chroma cosine", "Blues", (0, 1)),
                                                    (vol[:-2].mean(0), "learned heads (mean)", "RdBu_r", (-1, 1)))):
            ax = axes[i, j]
            ax.imshow(img, aspect="auto", origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1],
                      extent=[-500, 500, 0, img.shape[0] * 0.04])
            ax.grid(False)
            if i == 0:
                ax.set_title(f"{c}\nP(singing)={float(d[c + '_p']):.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ttl}\ntime (s)", fontsize=8)
            if i == 1:
                ax.set_xlabel("lag: song frame - mic frame (ms)", fontsize=8)
    fig.suptitle(f"Mic-by-song cross-similarity, one realistic dev window ({d['name']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "xsim_examples.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_roc, fig_cond_fp, fig_degradation, fig_ablation, fig_examples):
        f()
    print("figures ->", OUT)
