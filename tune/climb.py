"""Hill climb the sing-along classifier. Offline only; touches no audio devices.

Fixed track split: train (60%), dev (25%, used for every comparison), holdout (15%, read once
at the end). Objective: share of singing windows caught on dev, with the threshold set so
<=2% of non-singing windows on train (out-of-fold) are called singing. Every run is appended
to tune/results.jsonl; each step keeps a change only if dev improves.

Usage: python climb.py            run the climb
       python climb.py --final    also score the best config on the holdout split
       python climb.py --export   fit best_config.json on train+dev, write ../sing_model.joblib
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import librosa
import musdb
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import tune as T  # noqa: E402  (data loading + mic simulation)
from melody import window_features as _wf  # noqa: E402


def window_features(voice, song, cfg):
    return _wf(voice, song, cfg.tracker, cfg.contour)

SR = 16_000
HOP = 320
STEP = 0.75  # service evaluates every 0.75 s
MAX_FP = 0.02
CACHE = ROOT / "data" / "climb_signals.npz"
FULL_CACHE = ROOT / "data" / "climb_signals_full.npz"
FULL_CHUNKS, FULL_CHUNK_SEC = 3, 15  # per full-length song: 3 stretches of 15 s with vocals
RESULTS = HERE / "results.jsonl"
CONDS = ("sung", "talk", "other_song", "bleed")


@dataclass(frozen=True)
class Config:
    window: float = 1.5          # seconds of audio per decision
    smooth: int = 1              # average of the last N window probabilities
    tracker: str = "yin"         # yin | pyin
    contour: bool = False        # add melody-contour-following features
    model: str = "logreg"        # logreg | gbm
    hop_mult: int = 1            # 1 = windows every 0.75 s; 2 = every 0.375 s (more training data)
    # gradient-boosting hyperparameters (tuned by hparam.py)
    lr: float = 0.05
    depth: int = 3
    iters: int = 300
    min_leaf: int = 20
    l2: float = 0.0


# ---------- signals ----------

def build_signals() -> dict:
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["signals"].item()
    db = musdb.DB(root=str(ROOT / "data" / "musdb"), download=False)
    speech = [c for cl in T.load_speech(ROOT / "data" / "libri" / "LibriSpeech" / "dev-clean").values() for c in cl]
    tracks = [(t.name, T.to16k(t.audio, t.rate), T.to16k(t.targets["vocals"].audio, t.rate)) for t in db]
    sig = {}
    for i, (name, mix, voc) in enumerate(tracks):
        n = mix.size
        lag = int(T.rng.integers(0, int(0.2 * SR)))
        sung = np.concatenate([np.zeros(lag, np.float32), voc])[:n]
        talk = np.concatenate([speech[int(T.rng.integers(len(speech)))] for _ in range(3)])[:n]
        other = tracks[(i + 1 + int(T.rng.integers(len(tracks) - 1))) % len(tracks)][2]
        other = np.pad(other, (0, max(0, n - other.size)))[:n]
        sig[i] = {"name": name, "mix": mix, "voc": voc,
                  "sung": T.as_mic(sung, mix), "talk": T.as_mic(np.pad(talk, (0, max(0, n - talk.size))), mix),
                  "other_song": T.as_mic(other, mix), "bleed": T.as_mic(mix)}
        print(f"\r  signals {i + 1}/{len(tracks)}", end="", flush=True)
    print()
    np.savez(CACHE, signals=np.array(sig, dtype=object))
    return sig


def build_full_signals() -> dict:
    """Full MUSDB18 (150 songs): FULL_CHUNKS vocal-heavy stretches per song, one entry per
    stretch, tagged with its song so splits and folds never separate a song's stretches."""
    if FULL_CACHE.exists():
        return np.load(FULL_CACHE, allow_pickle=True)["signals"].item()
    db = musdb.DB(root=str(ROOT / "data" / "musdb18"), download=False)
    speech = [c for cl in T.load_speech(ROOT / "data" / "libri" / "LibriSpeech" / "dev-clean").values() for c in cl]
    L = FULL_CHUNK_SEC * SR
    sig, prev_voc = {}, []
    for song, t in enumerate(db):
        mix, voc = T.to16k(t.audio, t.rate), T.to16k(t.targets["vocals"].audio, t.rate)
        lag = int(T.rng.integers(0, int(0.2 * SR)))
        sung_full = np.concatenate([np.zeros(lag, np.float32), voc])[:mix.size]
        # rank 15 s stretches by how much of them has vocals
        starts = np.arange(0, mix.size - L, L)
        active = [np.mean(librosa.feature.rms(y=voc[a:a + L], hop_length=SR)[0] > 0.02) for a in starts]
        for ci, a in enumerate(starts[np.argsort(active)[::-1][:FULL_CHUNKS]]):
            m = mix[a:a + L]
            talk = np.concatenate([speech[int(T.rng.integers(len(speech)))] for _ in range(4)])[:L]
            other = prev_voc[int(T.rng.integers(len(prev_voc)))] if prev_voc else voc[(a + L) % (mix.size - L):][:L]
            other = np.pad(other, (0, L - other.size))
            sig[len(sig)] = {k: v.astype(np.float16) if isinstance(v, np.ndarray) else v for k, v in {
                "name": t.name, "song": song, "chunk": ci, "mix": m, "voc": voc[a:a + L],
                "sung": T.as_mic(sung_full[a:a + L], m), "talk": T.as_mic(np.pad(talk, (0, L - talk.size)), m),
                "other_song": T.as_mic(other, m), "bleed": T.as_mic(m)}.items()}
            prev_voc.append(voc[a:a + L])
        prev_voc = prev_voc[-30:]
        print(f"\r  full signals {song + 1}/{len(db)} songs", end="", flush=True)
    print()
    np.savez(FULL_CACHE, signals=np.array(sig, dtype=object))
    return sig


def n_songs(sig: dict) -> int:
    return len({s.get("song", i) for i, s in sig.items()})


def split(n_tracks: int) -> dict[int, str]:
    order = np.random.default_rng(42).permutation(n_tracks)
    a, b = int(0.6 * n_tracks), int(0.85 * n_tracks)
    return {int(t): ("train" if k < a else "dev" if k < b else "holdout") for k, t in enumerate(order)}


# ---------- features ----------

def featurize(sig: dict, cfg: Config):
    w = int(cfg.window * SR)
    step = int(STEP * SR / cfg.hop_mult)

    def one(i):
        s = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in sig[i].items()}
        song, k0 = s.get("song", i), 1000 * s.get("chunk", 0)  # k0 keeps a song's stretches apart
        rows = []
        for cond in CONDS:
            for k, start in enumerate(range(0, s["mix"].size - w + 1, step)):
                if cond == "sung" and np.sqrt(np.mean(s["voc"][start:start + w] ** 2)) < 0.02:
                    continue  # no vocals in this stretch
                f = window_features(s[cond][start:start + w], s["mix"][start:start + w], cfg)
                if f is not None:
                    rows.append((f, int(cond == "sung"), song, CONDS.index(cond), k0 + k))
        return rows

    rows = [r for rs in Parallel(n_jobs=-1)(delayed(one)(i) for i in sig) for r in rs]
    X = np.stack([r[0] for r in rows])
    y, g, c, k = (np.array([r[j] for r in rows]) for j in range(1, 5))
    return X, y, g, c, k


_feat_cache: dict[tuple, tuple] = {}


def get_features(sig, cfg):
    key = (cfg.window, cfg.tracker, cfg.contour, cfg.hop_mult)
    if key not in _feat_cache:
        _feat_cache[key] = featurize(sig, cfg)
    return _feat_cache[key]


# ---------- evaluation ----------

def make_model(cfg: Config):
    if cfg.model == "gbm":
        return HistGradientBoostingClassifier(max_depth=cfg.depth, learning_rate=cfg.lr, max_iter=cfg.iters,
                                              min_samples_leaf=cfg.min_leaf, l2_regularization=cfg.l2,
                                              class_weight="balanced", random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def smooth(p, g, c, k, n, hop_mult):
    """Mean of the last n decisions (0.75 s apart) within the same track/condition stream."""
    if n == 1:
        return p
    out = p.copy()
    idx = {(a, b, d): j for j, (a, b, d) in enumerate(zip(g, c, k))}
    for j, (a, b, d) in enumerate(zip(g, c, k)):
        prev = [idx.get((a, b, d - m * hop_mult)) for m in range(n)]
        out[j] = np.mean([p[q] for q in prev if q is not None])
    return out


def evaluate(sig, cfg: Config, parts: dict[int, str], on: str = "dev") -> dict:
    X, y, g, c, k = get_features(sig, cfg)
    part = np.array([parts[t] for t in g])
    tr, te = part == "train", part == on
    # out-of-fold train probabilities set the threshold, so it isn't fit to memorized data
    oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(X[tr], y[tr], g[tr]):
        oof[b] = make_model(cfg).fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
    oof = smooth(oof, g[tr], c[tr], k[tr], cfg.smooth, cfg.hop_mult)
    thr = float(np.quantile(oof[y[tr] == 0], 1 - MAX_FP))
    model = make_model(cfg).fit(X[tr], y[tr])
    p = smooth(model.predict_proba(X[te])[:, 1], g[te], c[te], k[te], cfg.smooth, cfg.hop_mult)
    hit = p >= thr
    neg = y[te] == 0
    return {"caught": float(hit[~neg].mean()), "false_singing": float(hit[neg].mean()),
            "per_cond_fp": {CONDS[ci]: float(hit[neg & (c[te] == ci)].mean()) for ci in range(1, 4)},
            "threshold": thr, "n": int(te.sum()), "_model": model}


def log(cfg: Config, res: dict, note: str, split_name: str = "dev") -> None:
    row = {"t": time.strftime("%H:%M:%S"), "split": split_name, "note": note, **asdict(cfg),
           **{k: v for k, v in res.items() if not k.startswith("_")}}
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  [{split_name}] {note:28s} caught={res['caught']:.1%}  false={res['false_singing']:.1%}  "
          f"fp by cond={ {k: round(v, 3) for k, v in res['per_cond_fp'].items()} }", flush=True)


# ---------- climb ----------

STEPS = [  # (note, change) in priority order; each kept only if dev "caught" improves
    ("window 3.0s", {"window": 3.0}),
    ("window 2.25s", {"window": 2.25}),
    ("smooth 2", {"smooth": 2}),
    ("smooth 3", {"smooth": 3}),
    ("smooth 4", {"smooth": 4}),
    ("pyin tracker", {"tracker": "pyin"}),
    ("contour features", {"contour": True}),
    ("2x overlapping windows", {"hop_mult": 2}),
    ("gradient boosting", {"model": "gbm"}),
]


def export(sig, cfg: Config, parts: dict[int, str]) -> None:
    """Fit cfg on train+dev (holdout stays unseen) and save the bundle the service loads."""
    X, y, g, c, k = get_features(sig, cfg)
    use = np.array([parts[t] != "holdout" for t in g])
    X, y, g, c, k = X[use], y[use], g[use], c[use], k[use]
    oof = np.zeros(y.size)
    for a, b in GroupKFold(5).split(X, y, g):
        oof[b] = make_model(cfg).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    oof = smooth(oof, g, c, k, cfg.smooth, cfg.hop_mult)
    thr = float(np.quantile(oof[y == 0], 1 - MAX_FP))
    joblib.dump({**asdict(cfg), "model_type": cfg.model, "model": make_model(cfg).fit(X, y), "threshold": thr},
                ROOT / "sing_model.joblib")
    print(f"exported {cfg} threshold={thr:.3f} -> sing_model.joblib")


def main() -> None:
    sig = build_signals()
    parts = split(n_songs(sig))
    if "--export" in sys.argv:
        export(sig, Config(**json.loads((HERE / "best_config.json").read_text())), parts)
        return
    best = Config()
    best_res = evaluate(sig, best, parts)
    log(best, best_res, "baseline")
    for note, change in STEPS:
        cand = replace(best, **change)
        if change.get("smooth") and cand.smooth <= best.smooth and best.smooth > 1:
            continue
        res = evaluate(sig, cand, parts)
        log(cand, res, note)
        if res["caught"] > best_res["caught"] + 0.005:
            best, best_res = cand, res
            print(f"    -> kept")
    print(f"\nbest: {best}  dev caught={best_res['caught']:.1%}")
    (HERE / "best_config.json").write_text(json.dumps(asdict(best), indent=2))
    if "--final" in sys.argv:
        final = evaluate(sig, best, parts, on="holdout")
        log(best, final, "FINAL", "holdout")


if __name__ == "__main__":
    main()
