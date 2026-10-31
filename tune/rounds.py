"""Feature rounds on the realistic benchmark, from the raw frame cache (rawcache.py). Offline only.

Same protocol as gbm_real.py: GBM fit on realistic train, threshold = 2% quantile of out-of-fold
train non-singing probabilities (3-decision smoothing), score = share of singing caught on dev.
Bootstrap CIs resample songs. Every run appends to results.jsonl.

Usage: python rounds.py <name> [<name> ...]     run named experiments from EXPERIMENTS (cached frames)
       python rounds.py --live final              same, on features computed like the service (slow, cached)
       python rounds.py --export                  fit `final` on live features (train+dev), write ../sing_model.joblib
       python rounds.py --holdout <name>        score one experiment on holdout (once)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import climb  # noqa: E402
import evalkit as E  # noqa: E402
from augment import CONDS_REAL  # noqa: E402
from melody import frame_features, frames  # noqa: E402

SR, HOP = 16_000, 320
STEP = int(0.75 * SR)
RAW = ROOT / "data" / "raw_realistic.npz"
WINDOWS = (3.0, 4.0)


@dataclass(frozen=True)
class Exp:
    window: float = 3.0
    groups: tuple = ("base",)
    lag: int = 15          # +- frames for the alignment search
    lag_step: int = 3
    smooth: int = 3
    lr: float = 0.05
    depth: int = 3
    iters: int = 300
    min_leaf: int = 20
    l2: float = 0.0
    extra: dict = field(default_factory=dict, hash=False, compare=False)


feats = frame_features  # shared with the live service (melody.py)


# ---------- dataset ----------

_raw = None


def raw() -> dict:
    global _raw
    if _raw is None:
        _raw = np.load(RAW, allow_pickle=True)["raw"].item()
    return _raw


def windows_of(r: dict, window: float):
    """Yield (cond_index, k, frame dict) for every window in a chunk."""
    nf = int(window * SR / HOP)
    wi = WINDOWS.index(window)
    starts = list(range(0, 15 * SR - int(window * SR) + 1, STEP))
    vr = r["voc_rms"]
    offs = sum(len(range(0, 15 * SR - int(w * SR) + 1, STEP)) for w in WINDOWS[:wi])
    for ci, cond in enumerate(CONDS_REAL):
        if cond == "shifted" and not r["shift_ok"]:
            continue
        d = r["conds"][cond]
        for k, a in enumerate(starts):
            if cond == "sung" and vr[offs + k] < 0.02:
                continue
            f = int(round(a / HOP))
            sl = slice(f, f + nf)
            yield ci, k, {"f0": d["f0"][sl], "vp": d["vp"][sl], "rms": d["rms"][sl], "mic_cq": d["cq"][:, sl],
                          "mic_on": d["on"][sl], "song_st": r["song_st"][:, sl], "song_cq": r["song_cq"][:, sl],
                          "song_on": r["song_on"][sl], "leak": d["leak"][window][k]}


def _chunk_rows(r, e):
    k0 = 1000 * r["chunk"]
    return [(f, int(ci == 0), r["song"], ci, k0 + k) for ci, k, w in windows_of(r, e.window)
            for f in [feats(w, e)] if f is not None]


def dataset(e: Exp):
    R = raw()
    out = Parallel(n_jobs=-1)(delayed(_chunk_rows)(R[i], e) for i in sorted(R))
    rows = [x for rs in out for x in rs]
    X = np.stack([x[0] for x in rows])
    y, g, c, k = (np.array([x[j] for x in rows]) for j in range(1, 5))
    return X, y, g, c, k


LIVE = ROOT / "data" / "feats_live_{name}.npz"


def _live_rows(s, e):
    """Features exactly as the service computes them: frames() on each isolated window."""
    s = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in s.items()}
    W, k0, rows = int(e.window * SR), 1000 * s["chunk"], []
    for ci, cond in enumerate(CONDS_REAL):
        if cond == "shifted" and not s.get("shift_ok", True):
            continue
        for k, a in enumerate(range(0, s["mix"].size - W + 1, STEP)):
            if cond == "sung" and np.sqrt(np.mean(s["voc"][a:a + W] ** 2)) < 0.02:
                continue
            f = frame_features(frames(s[cond][a:a + W], s["mix"][a:a + W]), e)
            if f is not None:
                rows.append((f, int(ci == 0), s["song"], ci, k0 + k))
    return rows


def dataset_live(name: str, e: Exp):
    path = Path(str(LIVE).format(name=name))
    if path.exists():
        d = np.load(path)
        return tuple(d[k] for k in ("X", "y", "g", "c", "k"))
    import warnings
    from augment import load_realistic
    warnings.filterwarnings("ignore")
    sig = load_realistic()
    out = Parallel(n_jobs=-1, verbose=2)(delayed(_live_rows)(sig[i], e) for i in sorted(sig))
    rows = [x for rs in out for x in rs]
    X = np.stack([x[0] for x in rows])
    y, g, c, k = (np.array([x[j] for x in rows]) for j in range(1, 5))
    np.savez(path, X=X, y=y, g=g, c=c, k=k)
    return X, y, g, c, k


# ---------- scoring ----------

def model(e: Exp, seed: int = 0):
    return HistGradientBoostingClassifier(max_depth=e.depth, learning_rate=e.lr, max_iter=e.iters,
                                          min_samples_leaf=e.min_leaf, l2_regularization=e.l2,
                                          class_weight="balanced", random_state=seed)


def run(name: str, e: Exp, on: str = "dev", seeds=(0, 1, 2), live: bool = False) -> dict:
    X, y, g, c, k = dataset_live(name.split("@")[0], e) if live else dataset(e)
    parts = climb.split(150)
    part = np.array([parts[t] for t in g])
    tr, te = part == "train", part == on
    res_all = []
    for seed in seeds:
        oof = np.zeros(tr.sum())
        for a, b in GroupKFold(5).split(X[tr], y[tr], g[tr]):
            oof[b] = model(e, seed).fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
        thr = E.threshold(E.smooth(oof, g[tr], c[tr], k[tr], e.smooth), y[tr])
        m = model(e, seed).fit(X[tr], y[tr])
        p = E.smooth(m.predict_proba(X[te])[:, 1], g[te], c[te], k[te], e.smooth)
        res = E.metrics(p, y[te], c[te], thr, CONDS_REAL)
        res.update(E.bootstrap(p, y[te], g[te], thr))
        E.save_preds(f"round_{name}_{on}_s{seed}", p=p, y=y[te], g=g[te], c=c[te], k=k[te], thr=thr)
        row = {"split": on, "eval": "realistic", "train_data": "realistic", "seed": seed,
               "note": f"round {name}", "exp": name, **{f: getattr(e, f) for f in
                                                        ("window", "groups", "lag", "lag_step", "smooth",
                                                         "lr", "depth", "iters", "min_leaf", "l2")}}
        row["groups"] = list(e.groups)
        E.log(row, res)
        res_all.append(res)
    mean = {key: float(np.mean([r[key] for r in res_all])) for key in ("caught", "false_singing")}
    print(f"== {name}: caught {mean['caught']:.1%}  false {mean['false_singing']:.1%}  (mean of {len(seeds)} seeds)")
    return mean


B = ("base",)
EXPERIMENTS = {
    "r1_baseline": Exp(),
    "r2_lag": Exp(lag=25, lag_step=1),
    "r3_contrast": Exp(lag=25, lag_step=1, groups=B + ("contrast",)),
    "r4_soft": Exp(lag=25, lag_step=1, groups=B + ("contrast", "soft")),
    "r5_vchroma": Exp(lag=25, lag_step=1, groups=B + ("contrast", "soft", "vchroma")),
    "r6_onset": Exp(lag=25, lag_step=1, groups=B + ("contrast", "soft", "vchroma", "onset")),
    "r7_extras": Exp(lag=25, lag_step=1, groups=B + ("contrast", "soft", "vchroma", "onset", "lagstats", "vp")),
    # best on dev (75.2% caught / 2.5% false); holdout 64.7% / 1.6% vs. 48.8% / 1.5% for r1_baseline
    "final": Exp(window=4.0, smooth=4, lag=25, lag_step=1, extra={"no_lagpos": True},
                 groups=B + ("contrast", "soft", "vchroma", "onset", "lagstats", "vp")),
}


def export(e: Exp) -> None:
    """Fit e on train+dev (holdout unseen), OOF threshold, write ../sing_model.joblib for the service.
    Trained on live-path features (dataset_live), so the service sees exactly what the model saw."""
    import joblib
    X, y, g, c, k = dataset_live("final", e)
    parts = climb.split(150)
    use = np.array([parts[t] != "holdout" for t in g])
    X, y, g, c, k = X[use], y[use], g[use], c[use], k[use]
    oof = np.zeros(y.size)
    for a, b in GroupKFold(5).split(X, y, g):
        oof[b] = model(e).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    thr = E.threshold(E.smooth(oof, g, c, k, e.smooth), y)
    joblib.dump({"version": 2, "window": e.window, "smooth": e.smooth, "groups": list(e.groups), "lag": e.lag,
                 "lag_step": e.lag_step, "extra": dict(e.extra), "model": model(e).fit(X, y), "threshold": thr},
                ROOT / "sing_model.joblib")
    print(f"exported {e} threshold={thr:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--export":
        export(EXPERIMENTS["final"])
        sys.exit()
    on = "dev"
    if args and args[0] == "--holdout":
        on, args = "holdout", args[1:]
    live = "--live" in args
    for a in [a for a in args if a != "--live"]:
        run(a + ("@live" if live else ""), EXPERIMENTS[a], on=on, live=live)
