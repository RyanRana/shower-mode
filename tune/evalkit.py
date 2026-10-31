"""Shared scoring for the clean/realistic benchmarks: same rules as climb.py.

Threshold: 2% quantile of out-of-fold *train* non-singing probabilities (after smoothing).
Score: share of singing windows caught on the test split; false singing = share of
non-singing windows called singing. Bootstrap CIs resample *tracks* (songs), not windows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
PREDS = HERE / "preds"
MAX_FP = 0.02
STEP = 0.75


def smooth(p, g, c, k, n: int):
    """Mean of the last n decisions in the same (song, condition, chunk-step) stream (climb.smooth)."""
    if n == 1:
        return p
    out = p.copy()
    idx = {(a, b, d): j for j, (a, b, d) in enumerate(zip(g, c, k))}
    for j, (a, b, d) in enumerate(zip(g, c, k)):
        prev = [idx.get((a, b, d - m)) for m in range(n)]
        out[j] = np.mean([p[q] for q in prev if q is not None])
    return out


def threshold(p_oof, y) -> float:
    return float(np.quantile(p_oof[y == 0], 1 - MAX_FP))


def metrics(p, y, c, thr, conds) -> dict:
    hit = p >= thr
    neg = y == 0
    return {"caught": float(hit[~neg].mean()), "false_singing": float(hit[neg].mean()),
            "per_cond_fp": {conds[ci]: float(hit[neg & (c == ci)].mean())
                            for ci in range(1, len(conds)) if (neg & (c == ci)).any()},
            "threshold": float(thr), "n": int(len(p))}


def bootstrap(p, y, g, thr, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of caught and false singing, resampling songs with replacement."""
    rng = np.random.default_rng(seed)
    songs = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in songs}
    hit = p >= thr
    cs, fs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([by[s] for s in rng.choice(songs, songs.size)])
        yy, hh = y[idx], hit[idx]
        cs.append(hh[yy == 1].mean())
        fs.append(hh[yy == 0].mean())
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"caught_ci": q(cs), "false_ci": q(fs)}


def log(row: dict, res: dict) -> None:
    out = {"t": time.strftime("%H:%M:%S"), **row, **{k: v for k, v in res.items() if not k.startswith("_")}}
    with RESULTS.open("a") as f:
        f.write(json.dumps(out) + "\n")
    ci = f"  CI {res['caught_ci'][0]:.1%}-{res['caught_ci'][1]:.1%}" if "caught_ci" in res else ""
    print(f"  [{row.get('split')}/{row.get('eval')}] {row.get('note', '')[:44]:44s} caught={res['caught']:.1%}{ci}  "
          f"false={res['false_singing']:.1%}  fp={ {k: round(v, 3) for k, v in res['per_cond_fp'].items()} }",
          flush=True)


def save_preds(name: str, **arrays) -> None:
    PREDS.mkdir(exist_ok=True)
    np.savez_compressed(PREDS / f"{name}.npz", **arrays)
