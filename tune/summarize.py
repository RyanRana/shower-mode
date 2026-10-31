"""Summary table for the realistic-benchmark round: mean +- sd over seeds, track-bootstrap 95% CI
(seed 0), per-condition false singing. Reads tune/results.jsonl.

Usage: python summarize.py [dev|holdout]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def key(r):
    if r.get("model") == "xsim":
        return f"xsim {r['variant']}"
    if r.get("model") == "gbm" and "train_data" in r:
        return f"GBM {r['train_data']}-trained"
    return None


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    rows = [json.loads(x) for x in (HERE / "results.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("split") == split and key(r) and "eval" in r]
    groups: dict[tuple, dict[int, dict]] = {}
    for r in rows:  # last run per (model, eval, seed) wins
        groups.setdefault((key(r), r["eval"]), {})[r.get("seed", 0)] = r
    print(f"| model | eval | seeds | caught (mean+-sd) | 95% CI (tracks, seed 0) | false singing | per-condition FP |")
    print("|---|---|---|---|---|---|---|")
    for (name, ev), by_seed in sorted(groups.items()):
        rs = list(by_seed.values())
        c = np.array([r["caught"] for r in rs])
        f = np.array([r["false_singing"] for r in rs])
        r0 = by_seed.get(0, rs[0])
        ci = r0.get("caught_ci", [np.nan, np.nan])
        conds = r0["per_cond_fp"].keys()
        pc = ", ".join(f"{k} {np.mean([r['per_cond_fp'][k] for r in rs]):.1%}" for k in conds)
        print(f"| {name} | {ev} | {len(rs)} | {c.mean():.1%} +- {c.std():.1%} | {ci[0]:.1%}-{ci[1]:.1%} | "
              f"{f.mean():.1%} +- {f.std():.1%} | {pc} |")


if __name__ == "__main__":
    main()
