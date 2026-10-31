"""Hyperparameter search for the sing-along classifier on full MUSDB18. Offline only.

Searches window length, smoothing, and gradient-boosting settings with Optuna (TPE).
Objective: share of singing caught on dev at the climb.py threshold rule (<=2% of
non-singing called singing, set from out-of-fold train predictions). The current best
config is scored on the same data as a baseline; holdout is scored once at the end
for the baseline and the winner. Writes best_config.json only if the winner beats the
baseline on dev.

Usage: python hparam.py [n_trials] [--preview]   (--preview: 7 s MUSDB previews instead of full songs)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import climb  # noqa: E402

WINDOWS = (2.25, 3.0, 4.0)
DATA = "preview" if "--preview" in sys.argv else "full"


def cached_features(sig, cfg):
    """climb.get_features, persisted to disk per window (pyin over the full set is the slow part)."""
    path = climb.ROOT / "data" / f"feats_{DATA}_w{cfg.window}_{cfg.tracker}.npz"
    key = (cfg.window, cfg.tracker, cfg.contour, cfg.hop_mult)
    if key not in climb._feat_cache:
        if path.exists():
            d = np.load(path)
            climb._feat_cache[key] = tuple(d[k] for k in ("X", "y", "g", "c", "k"))
        else:
            out = climb.featurize(sig, cfg)
            np.savez(path, **dict(zip(("X", "y", "g", "c", "k"), out)))
            climb._feat_cache[key] = out
    return climb._feat_cache[key]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_trials = int(args[0]) if args else 60
    sig = climb.build_full_signals() if DATA == "full" else climb.build_signals()
    parts = climb.split(climb.n_songs(sig))
    climb.get_features = cached_features
    for w in WINDOWS:
        print(f"features for window {w}s...", flush=True)
        cached_features(sig, climb.Config(window=w, tracker="pyin"))

    base = climb.Config(**json.loads((HERE / "best_config.json").read_text()))
    base_res = climb.evaluate(sig, base, parts)
    climb.log(base, base_res, f"{DATA}: current best")

    def objective(trial: optuna.Trial) -> float:
        cfg = replace(base,
                      window=trial.suggest_categorical("window", WINDOWS),
                      smooth=trial.suggest_int("smooth", 1, 5),
                      lr=trial.suggest_float("lr", 0.01, 0.3, log=True),
                      depth=trial.suggest_int("depth", 2, 8),
                      iters=trial.suggest_int("iters", 100, 800, step=50),
                      min_leaf=trial.suggest_int("min_leaf", 5, 100, log=True),
                      l2=trial.suggest_float("l2", 1e-4, 10, log=True))
        res = climb.evaluate(sig, cfg, parts)
        trial.set_user_attr("false_singing", res["false_singing"])
        climb.log(cfg, res, f"{DATA} optuna trial {trial.number}")
        return res["caught"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({"window": base.window, "smooth": base.smooth, "lr": base.lr, "depth": base.depth,
                         "iters": base.iters, "min_leaf": base.min_leaf, "l2": 1e-4})
    study.optimize(objective, n_trials=n_trials)

    best = replace(base, **study.best_params)
    print(f"\nbaseline dev caught={base_res['caught']:.1%}   best dev caught={study.best_value:.1%}")
    print(f"best: {best}")
    final = {}
    for name, cfg in (("baseline", base), ("tuned", best)):
        final[name] = climb.evaluate(sig, cfg, parts, on="holdout")
        climb.log(cfg, final[name], f"FINAL {DATA} {name}", "holdout")
    # 60 trials can overfit dev, so the holdout must confirm the gain without more false singing
    b, t = final["baseline"], final["tuned"]
    if t["caught"] > b["caught"] + 0.02 and t["false_singing"] <= b["false_singing"] + 0.002:
        (HERE / "best_config.json").write_text(json.dumps(asdict(best), indent=2))
        print("best_config.json updated")
    else:
        print("holdout did not confirm the dev gain; best_config.json unchanged")


if __name__ == "__main__":
    main()
