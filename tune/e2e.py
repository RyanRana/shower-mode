"""Train and score the learned (MERT + attention head) sing-along detector.

Same data, track split, 2% false-singing threshold rule, and 3-decision smoothing as
climb.py, so its numbers compare directly with results.jsonl. Offline only.

Usage: python e2e.py            train on train, score dev
       python e2e.py --final    also score holdout (once)
       python e2e.py --export   train on train+dev, save ../sing_e2e.pt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import climb  # noqa: E402
from mert_sing import MertEncoder, SingHead  # noqa: E402

SR = 16_000
WIN = 3 * SR
EVAL_HOP = int(0.75 * SR)
TRAIN_HOP = int(0.25 * SR)
SMOOTH = 3
CACHE = ROOT / "data" / "e2e_feats.npz"
EPOCHS, BATCH, LR = 10, 64, 3e-4
MISMATCH = 1      # wrong-song negatives per positive
HEAD_LAYERS = 2
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)


def extract(sig, parts):
    """MERT features per window. Song windows are shared by the 4 mic conditions."""
    if CACHE.exists():
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    enc = MertEncoder()
    mic, song, meta = [], [], []  # meta: track, cond, k(0.75 s step index or -1), song_row, label
    song_rows: dict[tuple[int, int], int] = {}
    pend_mic, pend_song = [], []

    def flush():
        if pend_mic:
            mic.extend(enc([w for w in pend_mic]))
            pend_mic.clear()
        if pend_song:
            song.extend(enc([w for w in pend_song]))
            pend_song.clear()

    for n, i in enumerate(sig):
        s = sig[i]
        hop = TRAIN_HOP if parts[i] == "train" else EVAL_HOP
        for start in range(0, s["mix"].size - WIN + 1, hop):
            k = start // EVAL_HOP if start % EVAL_HOP == 0 else -1
            for ci, cond in enumerate(climb.CONDS):
                if cond == "sung" and np.sqrt(np.mean(s["voc"][start:start + WIN] ** 2)) < 0.02:
                    continue
                if (i, start) not in song_rows:
                    song_rows[(i, start)] = len(song) + len(pend_song)
                    pend_song.append(s["mix"][start:start + WIN])
                pend_mic.append(s[cond][start:start + WIN])
                meta.append((i, ci, k, song_rows[(i, start)], int(cond == "sung")))
                if len(pend_mic) >= 32:
                    flush()
        flush()
        print(f"\r  MERT {n + 1}/{len(sig)} tracks, {len(meta)} windows", end="", flush=True)
    print()
    out = {"mic": np.stack(mic), "song": np.stack(song), "meta": np.array(meta)}
    np.savez(CACHE, **out)
    return out


def train(F, rows: np.ndarray, epochs: int = EPOCHS) -> SingHead:
    mic, song, meta = F["mic"], F["song"], F["meta"]
    head = SingHead(mic.shape[1], layers=HEAD_LAYERS).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.1)
    y = meta[rows, 4]
    # each positive also appears MISMATCH times as a negative (its voice over other tracks' songs)
    neg_per_pos = (y == 0).sum() / max(1, (y == 1).sum()) + MISMATCH
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg_per_pos), dtype=torch.float32, device=DEV))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * (len(rows) // BATCH + 1))
    head.train()
    for _ in range(epochs):
        for b in np.array_split(np.random.permutation(rows), len(rows) // BATCH + 1):
            song_idx, labels = meta[b, 3], meta[b, 4]
            # mismatched pairs: without them the head can score "is this singing?" from the voice
            # alone and never learn to compare voice with song
            pos = np.flatnonzero(labels == 1)
            orig = b
            for r in range(1, min(MISMATCH, pos.size - 1) + 1):
                shuf = np.roll(pos, r)
                ok = meta[orig[pos], 0] != meta[orig[shuf], 0]
                b = np.concatenate([b, orig[pos[ok]]])
                song_idx = np.concatenate([song_idx, meta[orig[shuf[ok]], 3]])
                labels = np.concatenate([labels, np.zeros(ok.sum(), dtype=labels.dtype)])
            v = torch.from_numpy(mic[b].astype(np.float32)).to(DEV)
            s = torch.from_numpy(song[song_idx].astype(np.float32)).to(DEV)
            # augmentation: random lag of the voice (people trail the artist) + frame dropout
            v = torch.roll(v, int(np.random.randint(-3, 6)), dims=1)
            v = v * (torch.rand(v.shape[:2], device=DEV) > 0.1).unsqueeze(-1)
            loss = loss_fn(head(v, s), torch.from_numpy(labels).float().to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return head.eval()


@torch.no_grad()
def predict(head: SingHead, F, rows: np.ndarray) -> np.ndarray:
    out = []
    for b in np.array_split(rows, max(1, len(rows) // 256)):
        v = torch.from_numpy(F["mic"][b].astype(np.float32)).to(DEV)
        s = torch.from_numpy(F["song"][F["meta"][b, 3]].astype(np.float32)).to(DEV)
        out.append(torch.sigmoid(head(v, s)).cpu().numpy())
    return np.concatenate(out)


def smoothed(p, meta_rows):
    g, c, k = meta_rows[:, 0], meta_rows[:, 1], meta_rows[:, 2]
    return climb.smooth(p, g, c, k, SMOOTH, 1)


def threshold_from_oof(F, rows) -> float:
    """5-fold by track: out-of-fold probabilities at eval-hop windows set the 2% threshold."""
    meta = F["meta"]
    tracks = np.unique(meta[rows, 0])
    folds = np.array_split(np.random.default_rng(0).permutation(tracks), 5)
    ps, ms = [], []
    for f in folds:
        held = np.isin(meta[rows, 0], f)
        head = train(F, rows[~held])
        ev = rows[held & (meta[rows, 2] >= 0)]  # score only the 0.75 s grid, like the service
        ps.append(smoothed(predict(head, F, ev), meta[ev]))
        ms.append(meta[ev])
    p, m = np.concatenate(ps), np.concatenate(ms)
    return float(np.quantile(p[m[:, 4] == 0], 1 - climb.MAX_FP))


def score(F, train_rows, test_rows, note, split_name) -> dict:
    meta = F["meta"]
    thr = threshold_from_oof(F, train_rows)
    head = train(F, train_rows)
    p = smoothed(predict(head, F, test_rows), meta[test_rows])
    y, c = meta[test_rows, 4], meta[test_rows, 1]
    hit = p >= thr
    res = {"caught": float(hit[y == 1].mean()), "false_singing": float(hit[y == 0].mean()),
           "per_cond_fp": {climb.CONDS[ci]: float(hit[(y == 0) & (c == ci)].mean()) for ci in range(1, 4)},
           "threshold": thr, "n": int(len(test_rows))}
    row = {"t": time.strftime("%H:%M:%S"), "split": split_name, "note": note, "model": "mert+attention", **res}
    with climb.RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  [{split_name}] {note:28s} caught={res['caught']:.1%}  false={res['false_singing']:.1%}  "
          f"fp by cond={ {k: round(v, 3) for k, v in res['per_cond_fp'].items()} }", flush=True)
    return res


def main() -> None:
    sig = climb.build_signals()
    parts = climb.split(len(sig))
    F = extract(sig, parts)
    meta = F["meta"]
    part = np.array([parts[t] for t in meta[:, 0]])
    rows = np.arange(len(meta))
    train_rows = rows[part == "train"]
    if "--export" in sys.argv:
        # train+dev windows; dev only has the 0.75 s grid, which is fine for training
        fit = rows[part != "holdout"]
        thr = threshold_from_oof(F, fit)
        head = train(F, fit)
        torch.save({"state": head.state_dict(), "frames": F["mic"].shape[1], "threshold": thr,
                    "smooth": SMOOTH}, ROOT / "sing_e2e.pt")
        print(f"exported threshold={thr:.3f} -> sing_e2e.pt")
        return
    score(F, train_rows, rows[part == "dev"], "e2e MERT+attention", "dev")
    if "--final" in sys.argv:
        score(F, train_rows, rows[part == "holdout"], "e2e FINAL", "holdout")


if __name__ == "__main__":
    main()
