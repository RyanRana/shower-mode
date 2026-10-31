"""Rescore the current best GBM (best_config.json) on full MUSDB18 dev, clean and realistic.

Features: melody.window_features (pyin) on climb's full clean set (climb_signals_full.npz) and on
the realistic set (augment.py --build). Two training regimes:
  clean-trained:      fit + OOF threshold on clean train, scored on clean and realistic dev
  realistic-trained:  fit + OOF threshold on realistic train, scored on realistic and clean dev
Per-window dev predictions go to tune/preds/ for plots and bootstrap CIs.

Usage: python gbm_real.py                 (dev only)
       python gbm_real.py --final         (holdout only; run once, for the final step)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import climb  # noqa: E402
import evalkit as E  # noqa: E402
from augment import CONDS_REAL, load_realistic  # noqa: E402
from melody import window_features  # noqa: E402

SR = 16_000
CONDS_CLEAN = climb.CONDS
FEATS = {"clean": climb.ROOT / "data" / "feats_fullclean_w{w}_pyin.npz",
         "realistic": climb.ROOT / "data" / "feats_fullreal_w{w}_pyin.npz"}


def _entry_rows(s: dict, conds, w: int, step: int, tracker: str, contour: bool):
    s = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in s.items()}
    song, k0 = s["song"], 1000 * s["chunk"]
    rows = []
    for ci, cond in enumerate(conds):
        if cond == "shifted" and not s.get("shift_ok", True):
            continue
        for k, start in enumerate(range(0, s["mix"].size - w + 1, step)):
            if cond == "sung" and np.sqrt(np.mean(s["voc"][start:start + w] ** 2)) < 0.02:
                continue
            f = window_features(s[cond][start:start + w], s["mix"][start:start + w], tracker, contour)
            if f is not None:
                rows.append((f, int(cond == "sung"), song, ci, k0 + k))
    return rows


def features(kind: str, cfg) -> tuple:
    path = Path(str(FEATS[kind]).format(w=cfg.window))
    if path.exists():
        d = np.load(path)
        return tuple(d[k] for k in ("X", "y", "g", "c", "k"))
    sig = climb.build_full_signals() if kind == "clean" else load_realistic()
    conds = CONDS_CLEAN if kind == "clean" else CONDS_REAL
    w, step = int(cfg.window * SR), int(0.75 * SR)
    out = Parallel(n_jobs=-1, verbose=2)(delayed(_entry_rows)(sig[i], conds, w, step, cfg.tracker, cfg.contour)
                                         for i in sig)
    del sig
    rows = [r for rs in out for r in rs]
    X = np.stack([r[0] for r in rows])
    y, g, c, k = (np.array([r[j] for r in rows]) for j in range(1, 5))
    np.savez(path, X=X, y=y, g=g, c=c, k=k)
    return X, y, g, c, k


def fit_with_threshold(cfg, X, y, g, c, k, seed: int):
    make = lambda: climb.make_model(cfg).set_params(random_state=seed)  # noqa: E731
    oof = np.zeros(y.size)
    for a, b in GroupKFold(5).split(X, y, g):
        oof[b] = make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    oof = E.smooth(oof, g, c, k, cfg.smooth)
    return make().fit(X, y), E.threshold(oof, y)


def main() -> None:
    cfg = climb.Config(**json.loads((HERE / "best_config.json").read_text()))
    cfg = replace(cfg, model="gbm")
    on = ["holdout"] if "--final" in sys.argv else ["dev"]
    seeds = [0, 1, 2]
    data = {kind: features(kind, cfg) for kind in ("clean", "realistic")}
    parts = climb.split(150)
    conds = {"clean": CONDS_CLEAN, "realistic": CONDS_REAL}
    for train_kind in ("clean", "realistic"):
        X, y, g, c, k = data[train_kind]
        part = np.array([parts[t] for t in g])
        tr = part == "train"
        for seed in seeds:
            model, thr = fit_with_threshold(cfg, X[tr], y[tr], g[tr], c[tr], k[tr], seed)
            for split_name in on:
                for ev in ("clean", "realistic"):
                    Xe, ye, ge, ce, ke = data[ev]
                    pe = np.array([parts[t] for t in ge]) == split_name
                    p = E.smooth(model.predict_proba(Xe[pe])[:, 1], ge[pe], ce[pe], ke[pe], cfg.smooth)
                    res = E.metrics(p, ye[pe], ce[pe], thr, conds[ev])
                    res.update(E.bootstrap(p, ye[pe], ge[pe], thr))
                    name = f"gbm_{train_kind}train_{ev}_{split_name}_s{seed}"
                    E.save_preds(name, p=p, y=ye[pe], g=ge[pe], c=ce[pe], k=ke[pe], thr=thr,
                                 conds=np.array(conds[ev]))
                    E.log({"split": split_name, "eval": ev, "train_data": train_kind, "seed": seed,
                           "note": f"GBM best_config, {train_kind}-trained, full MUSDB18, {ev} eval",
                           **asdict(cfg)}, res)


if __name__ == "__main__":
    main()
