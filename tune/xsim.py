"""End-to-end cross-similarity sing-along detector (mic vs song mix). Offline only.

Both streams -> log-frequency spectrogram (16 kHz, 20 ms hop, 2 bins/semitone, 65 Hz-4.2 kHz)
-> a small learned conv encoder per stream -> frame embeddings. A mic-by-song cross-similarity
volume over lags +-500 ms (8 learned cosine heads + fixed chroma and spectral cosines) goes
through a 2D CNN; its pooled output plus each stream's pooled embedding -> P(singing along).

Trained on the fly (augment.py realistic pipeline, no feature cache) from the decoded MUSDB18
train songs with hard negatives: talk, other_song, bleed-only, shifted (own song's vocal >5 s
away). Threshold and scoring follow climb.py: 2% quantile of out-of-fold train non-singing
probabilities (5 folds by song as in climb.py, realistic train chunks), 3-decision smoothing, dev split.
Each run = 5 fold trainings + 1 full-train model, 350 steps x 128 crops each (~14 min on an M5).

Usage: python xsim.py --variant main --seed 0            train (5 OOF folds + full), score dev
       variants: main | noxsim (no cross-similarity) | weakbleed (clean-style bleed in training) | w3 (3 s window)
       python xsim.py --variant main --holdout              score holdout with the saved models (once)
       python xsim.py --examples                             dump example cross-similarity matrices for figs.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
import augment as A  # noqa: E402

SR, HOP = 16_000, 320
LAG = 25            # +-25 frames = +-500 ms
SMOOTH = 3
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = HERE / "models"
VARIANTS = {
    "main": dict(window=4.0, xsim=True, bleed="strong"),
    "noxsim": dict(window=4.0, xsim=False, bleed="strong"),
    "weakbleed": dict(window=4.0, xsim=True, bleed="weak"),
    "w3": dict(window=3.0, xsim=True, bleed="strong"),
}
COND_IDS = {c: i for i, c in enumerate(A.CONDS_REAL)}  # sung, talk, other_song, bleed, shifted


# ---------- on-the-fly training data ----------

class Gen:
    """(mic, song, label, valid, cond) crops: pick a train song and a vocal-heavy chunk, make all five
    conditions with fresh augmentation, take random crops. Run inside BatchPool workers."""

    def __init__(self, pool: list[int], window: float, bleed: str, seed: int, crops: int = 4,
                 extra: float = 4.0, buffer: int = 160) -> None:
        self.pool, self.W, self.bleed, self.seed = list(pool), int(window * SR), bleed, seed
        self.crops, self.C, self.buffer = crops, int((window + extra) * SR), buffer

    def _setup(self):
        S = A.Songs()
        self.S = S
        self.sp_audio = np.load(ROOT / "data" / "xsim_speech_audio.npy", mmap_mode="r")
        self.sp_offs = np.load(ROOT / "data" / "xsim_speech_offs.npy")
        fl = self.C // HOP
        self.cand = {}
        for i in self.pool:
            act = (S.rms(i) > 0.02).astype(np.float64)
            if act.size <= fl:
                continue
            c = np.concatenate([[0], np.cumsum(act)])
            frac = (c[fl:] - c[:-fl]) / fl
            starts = np.flatnonzero(frac[::25] >= 0.5) * 25 * HOP
            starts = starts[(starts >= 2 * SR) & (starts + self.C + 2 * SR < S.lens[i])]
            if starts.size:
                self.cand[i] = starts
        self.songs = list(self.cand)

    def _region(self, i, a, row, before=2, after=2):
        lo = a - before * SR
        x = self.S.audio[row, self.S.offs[i] + lo:self.S.offs[i] + a + self.C + after * SR].astype(np.float32) / 32767
        return x, before * SR

    def _talk(self, rng):
        n = self.C + 2 * SR
        parts, tot = [], 0
        while tot < n:
            j = int(rng.integers(len(self.sp_offs) - 1))
            parts.append(self.sp_audio[self.sp_offs[j]:self.sp_offs[j + 1]])
            tot += parts[-1].size
        return np.concatenate(parts)[:n].astype(np.float32) / 32767

    def _chunk(self, rng):
        C, W = self.C, self.W
        i = self.songs[int(rng.integers(len(self.songs)))]
        a = int(self.cand[i][int(rng.integers(self.cand[i].size))])
        voc, va = self._region(i, a, 1)
        mix, ma = self._region(i, a, 0, before=1, after=0)
        song = mix[ma:ma + C]
        dry = {"sung": A.augment_voice(voc, va, C, rng)}
        j = self.songs[int(rng.integers(len(self.songs)))]
        if j != i:
            b = int(self.cand[j][int(rng.integers(self.cand[j].size))])
            vj, vb = self._region(j, b, 1)
            dry["other_song"] = A.augment_voice(vj, vb, C, rng)
        b = A.pick_shift(a, C, int(self.S.lens[i]) - 2 * SR, self.S.chroma(i), self.S.rms(i), rng, tries=6)
        if b is not None and b > 2 * SR:
            vs, vsa = self._region(i, b, 1)
            dry["shifted"] = A.augment_voice(vs, vsa, C, rng)
        dry["talk"] = A.augment_voice(self._talk(rng), SR, C, rng, drops=False)
        dry["bleed"] = None
        out = []
        for cond, v in dry.items():
            if self.bleed == "weak":
                m = A.mic(v, song, rng, bleed="weak", song=song)
            else:
                m = A.mic(v, A.room_leak(mix, C, rng), rng)
            for o in rng.integers(0, C - W + 1, self.crops):
                y = int(cond == "sung")
                valid = 1
                if y and np.sqrt(np.mean(v[o:o + W] ** 2)) < 0.02:
                    valid = 0  # the singer is silent here: not a positive (and not a clean negative)
                out.append((m[o:o + W].astype(np.float16), song[o:o + W].astype(np.float16), y, valid,
                            COND_IDS[cond]))
        return out


# ---------- model ----------

def logfreq_fb(n_fft: int = 2048, lo: int = 36, bins: int = 144, per_semi: int = 2) -> torch.Tensor:
    f = np.linspace(0, SR / 2, n_fft // 2 + 1)
    df = f[1]
    fb = np.zeros((bins, f.size), np.float32)
    for k in range(bins):
        fc = 440 * 2 ** ((lo + k / per_semi - 69) / 12)
        width = max(fc * (2 ** (1 / (12 * per_semi)) - 1), df)
        fb[k] = np.maximum(0, 1 - np.abs(f - fc) / width)
        fb[k] /= fb[k].sum() + 1e-9
    return torch.from_numpy(fb)


class Front(nn.Module):
    """Log-frequency power spectrogram, per-window instance norm; also returns chroma."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("fb", logfreq_fb())
        self.register_buffer("win", torch.hann_window(2048))
        pcs = (np.round(36 + np.arange(144) / 2).astype(int)) % 12
        self.register_buffer("fold", torch.from_numpy(np.eye(12, dtype=np.float32)[pcs].T.copy()))  # (12, 144)

    def forward(self, x):
        X = torch.stft(x, 2048, HOP, window=self.win, center=True, return_complex=True)
        p = self.fb @ (X.real ** 2 + X.imag ** 2)                         # (B, 144, T)
        logp = torch.log10(p + 1e-9)
        mu = logp.mean(dim=(1, 2), keepdim=True)
        sd = logp.std(dim=(1, 2), keepdim=True) + 1e-5
        chroma = torch.sqrt(self.fold @ p + 1e-12)                        # (B, 12, T)
        return (logp - mu) / sd, chroma


def conv_bn(ci, co, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, stride, k // 2, bias=False), nn.BatchNorm2d(co), nn.GELU())


class Encoder(nn.Module):
    def __init__(self, D: int = 128) -> None:
        super().__init__()
        self.c2 = nn.Sequential(conv_bn(1, 16, stride=(2, 1)), conv_bn(16, 16, stride=(2, 1)))
        self.c1 = nn.Sequential(nn.Conv1d(16 * 36, D, 1), nn.GELU(),
                                nn.Conv1d(D, D, 5, padding=2, bias=False), nn.BatchNorm1d(D), nn.GELU(),
                                nn.Conv1d(D, D, 5, padding=2))

    def forward(self, s):                       # (B, 144, T) -> (B, D, T)
        h = self.c2(s.unsqueeze(1))
        return self.c1(h.flatten(1, 2))


def lag_sim(a, b, R: int = LAG):
    """a, b: (B, H, d, T) unit vectors along d -> (B, H, T, 2R+1), entry l = <a[t], b[t + l - R]>."""
    B, H, _, T = a.shape
    P = Fn.pad(a.transpose(-1, -2) @ b, (R, R))                     # (B, H, T, T+2R), zero outside
    W = T + 2 * R
    flat = Fn.pad(P.reshape(B, H, T * W), (0, T))                    # skew: row t starts at P[t, t]
    return flat.view(B, H, T, W + 1)[..., :2 * R + 1]


class XSim(nn.Module):
    def __init__(self, use_xsim: bool = True, D: int = 128, heads: int = 8) -> None:
        super().__init__()
        self.use_xsim, self.H = use_xsim, heads
        self.front = Front()
        self.enc_mic, self.enc_song = Encoder(D), Encoder(D)
        self.proj_mic, self.proj_song = nn.Conv1d(D, D, 1), nn.Conv1d(D, D, 1)
        if use_xsim:
            self.cnn = nn.Sequential(conv_bn(heads + 2, 24), conv_bn(24, 48, stride=2), conv_bn(48, 64))
        head_in = (128 if use_xsim else 0) + 3 * D
        self.head = nn.Sequential(nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def xsim(self, mic, song):
        """Cross-similarity volume (B, H+2, T, 2*LAG+1) and stream embeddings."""
        sm, cm = self.front(mic)
        ss, cs = self.front(song)
        em, es = self.enc_mic(sm), self.enc_song(ss)
        B, D, T = em.shape
        qm = Fn.normalize(self.proj_mic(em).view(B, self.H, D // self.H, T), dim=2)
        qs = Fn.normalize(self.proj_song(es).view(B, self.H, D // self.H, T), dim=2)
        fixed_c = lag_sim(Fn.normalize(cm, dim=1).unsqueeze(1), Fn.normalize(cs, dim=1).unsqueeze(1))
        zm, zs = sm - sm.mean(1, keepdim=True), ss - ss.mean(1, keepdim=True)
        fixed_s = lag_sim(Fn.normalize(zm, dim=1).unsqueeze(1), Fn.normalize(zs, dim=1).unsqueeze(1))
        vol = torch.cat([lag_sim(qm, qs), fixed_c, fixed_s], 1) if self.use_xsim else None
        if vol is not None:  # 20 ms frames -> 40 ms in time for the CNN (lags stay at 20 ms)
            vol = Fn.avg_pool2d(vol, (2, 1), ceil_mode=True)
        return vol, em, es

    def forward(self, mic, song):
        vol, em, es = self.xsim(mic, song) if self.use_xsim else (None, *self._emb(mic, song))
        feats = [em.mean(-1), em.amax(-1), es.mean(-1)]
        if self.use_xsim:
            h = self.cnn(vol).mean(2)                                    # mean over time -> (B, 64, L')
            feats += [h.amax(-1), h.mean(-1)]                             # best lag + average
        return self.head(torch.cat(feats, 1)).squeeze(-1)

    def _emb(self, mic, song):
        return self.enc_mic(self.front(mic)[0]), self.enc_song(self.front(song)[0])


# ---------- train / predict ----------

def _batch_worker(q, pool, window, bleed, seed, crops, batch, wid) -> None:
    """Worker process: fill a shuffle buffer from fresh chunks, push ready fp16 batches."""
    g = Gen(pool, window, bleed, seed, crops=crops)
    g._setup()
    rng = np.random.default_rng([seed, wid])
    buf: list = []
    while True:
        while len(buf) < 3 * batch:
            buf.extend(g._chunk(rng))
        rng.shuffle(buf)
        take, buf = buf[:batch], buf[batch:]
        q.put(tuple(np.stack([t[j] for t in take]) if j < 2 else np.array([t[j] for t in take])
                    for j in range(5)))


class BatchPool:
    """Spawned worker processes feeding a queue. Closed with terminate(), so teardown can never hang
    (torch's DataLoader teardown deadlocked once with an infinite IterableDataset)."""

    def __init__(self, pool, window, bleed, seed, crops, batch, workers) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=2 * workers)
        self.ps = [ctx.Process(target=_batch_worker, args=(self.q, pool, window, bleed, seed, crops, batch, w),
                               daemon=True) for w in range(workers)]
        for p in self.ps:
            p.start()

    def get(self):
        return self.q.get(timeout=600)

    def close(self) -> None:
        for p in self.ps:
            p.terminate()
        for p in self.ps:
            p.join(5)
        self.q.cancel_join_thread()
        self.q.close()


def train(pool, var: dict, seed: int, steps: int, batch: int = 128, workers: int = 4, lr: float = 2e-3,
          crops: int = 4):
    torch.manual_seed(seed)
    model = XSim(var["xsim"]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    bp = BatchPool(pool, var["window"], var["bleed"], seed * 1000 + len(pool), crops, batch, workers)
    model.train()
    t0, run = time.time(), []
    try:
        for step in range(steps):
            m, s, y, v, c = (torch.from_numpy(a) for a in bp.get())
            m, s, y, v = m.to(DEV).float(), s.to(DEV).float(), y.float().to(DEV), v.float().to(DEV)
            logit = model(m, s)
            npos, nneg = (y * v).sum().clamp(min=1), ((1 - y) * v).sum().clamp(min=1)
            w = v * torch.where(y > 0, 0.5 / npos, 0.5 / nneg)
            loss = (Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run.append(float(loss.detach()))
            if (step + 1) % 50 == 0:
                print(f"    step {step + 1}/{steps} loss {np.mean(run[-50:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        bp.close()
    return model.eval()


def windows(sig: dict, songs: set[int], W: int, conds) -> tuple:
    """Eval windows on the 0.75 s grid (climb.featurize rules), as index arrays into sig."""
    step = int(0.75 * SR)
    meta = []  # entry, cond, start, label, song, k
    for e, s in sig.items():
        if s["song"] not in songs:
            continue
        for ci, cond in enumerate(conds):
            if cond == "shifted" and not s.get("shift_ok", True):
                continue
            for k, st in enumerate(range(0, s["mix"].size - W + 1, step)):
                if cond == "sung" and np.sqrt(np.mean(s["voc"][st:st + W].astype(np.float32) ** 2)) < 0.02:
                    continue
                meta.append((e, ci, st, int(cond == "sung"), s["song"], 1000 * s["chunk"] + k))
    return np.array(meta, dtype=np.int64)


@torch.no_grad()
def predict(model, sig, meta, W, conds, batch: int = 256) -> np.ndarray:
    out = []
    for b in range(0, len(meta), batch):
        rows = meta[b:b + batch]
        m = np.stack([sig[e][conds[c]][st:st + W] for e, c, st, *_ in rows]).astype(np.float32)
        s = np.stack([sig[e]["mix"][st:st + W] for e, c, st, *_ in rows]).astype(np.float32)
        out.append(torch.sigmoid(model(torch.from_numpy(m).to(DEV), torch.from_numpy(s).to(DEV))).cpu().numpy())
    p = np.concatenate(out)
    import evalkit as E
    return E.smooth(p, meta[:, 4], meta[:, 1], meta[:, 5], SMOOTH)


def subset(sig: dict, songs: set[int]) -> dict:
    return {k: v for k, v in sig.items() if v["song"] in songs}


def main() -> None:
    import climb
    import evalkit as E
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()
    if args.examples:
        return dump_examples()
    var = VARIANTS[args.variant]
    W = int(var["window"] * SR)
    parts = climb.split(150)
    split_songs = {p: {i for i in range(150) if parts[i] == p} for p in ("train", "dev", "holdout")}
    tag = f"xsim_{args.variant}_s{args.seed}"
    note = (f"xsim {args.variant}: {'cross-sim CNN' if var['xsim'] else 'NO cross-sim (pooled embeddings only)'}, "
            f"{var['window']:.0f}s window, {var['bleed']} bleed train aug, {args.steps} steps, {args.folds}-fold OOF thr, {args.crops} crops/chunk")
    row = {"model": "xsim", "variant": args.variant, "seed": args.seed, "window": var["window"],
           "smooth": SMOOTH, "steps": args.steps, "folds": args.folds, "crops": args.crops,
           "train_data": "realistic-onthefly"}
    MODELS.mkdir(exist_ok=True)
    real = A.load_realistic()
    if args.holdout:
        ck = torch.load(MODELS / f"{tag}.pt", map_location=DEV)
        model = XSim(var["xsim"]).to(DEV)
        model.load_state_dict(ck["state"])
        model.eval()
        evaluate(model, ck["threshold"], real, climb.build_full_signals(), split_songs["holdout"], W, "holdout",
                 tag, note, row)
        return
    t0 = time.time()
    # out-of-fold threshold: --folds folds over train songs (fixed partition), scored on realistic train chunks
    tr_sig = subset(real, split_songs["train"])
    tr = np.array(sorted(split_songs["train"]))
    folds = np.array_split(np.random.default_rng(0).permutation(tr), args.folds)
    ps, ys = [], []
    for fi, fold in enumerate(folds):
        print(f"  fold {fi + 1}/{args.folds}", flush=True)
        model = train([int(s) for s in tr if s not in set(fold)], var, args.seed * 10 + fi, args.steps,
                      crops=args.crops)
        meta = windows(tr_sig, set(int(s) for s in fold), W, A.CONDS_REAL)
        ps.append(predict(model, tr_sig, meta, W, A.CONDS_REAL))
        ys.append(meta[:, 3])
    thr = E.threshold(np.concatenate(ps), np.concatenate(ys))
    print(f"  OOF threshold {thr:.4f}; full model", flush=True)
    model = train([int(s) for s in tr], var, args.seed * 10 + 9, args.steps, crops=args.crops)
    torch.save({"state": model.state_dict(), "threshold": thr, "variant": args.variant, **var},
               MODELS / f"{tag}.pt")
    row["train_minutes"] = round((time.time() - t0) / 60, 1)
    del tr_sig
    evaluate(model, thr, real, climb.build_full_signals(), split_songs["dev"], W, "dev", tag, note, row)


def evaluate(model, thr, real, clean, songs, W, split_name, tag, note, row) -> None:
    import climb
    import evalkit as E
    for ev, sig, conds in (("realistic", subset(real, songs), A.CONDS_REAL),
                           ("clean", subset(clean, songs), climb.CONDS)):
        meta = windows(sig, songs, W, conds)
        p = predict(model, sig, meta, W, conds)
        y, c, g = meta[:, 3], meta[:, 1], meta[:, 4]
        res = E.metrics(p, y, c, thr, conds)
        res.update(E.bootstrap(p, y, g, thr))
        E.save_preds(f"{tag}_{ev}_{split_name}", p=p, y=y, g=g, c=c, k=meta[:, 5], thr=thr, conds=np.array(conds))
        E.log({"split": split_name, "eval": ev, "note": note, **row}, res)


@torch.no_grad()
def dump_examples() -> None:
    """Cross-similarity (first learned head + fixed chroma channel) for one dev chunk, every condition."""
    import climb
    parts = climb.split(150)
    real = A.load_realistic()
    ck = torch.load(MODELS / "xsim_main_s0.pt", map_location=DEV)
    model = XSim(True).to(DEV)
    model.load_state_dict(ck["state"])
    model.eval()
    W = int(4.0 * SR)
    e = next(k for k, s in real.items() if parts[s["song"]] == "dev" and s["shift_ok"]
             and np.sqrt(np.mean(s["voc"][4 * SR:4 * SR + W].astype(np.float32) ** 2)) > 0.04)
    s = real[e]
    out = {}
    for cond in A.CONDS_REAL:
        m = torch.from_numpy(s[cond][4 * SR:4 * SR + W].astype(np.float32))[None].to(DEV)
        so = torch.from_numpy(s["mix"][4 * SR:4 * SR + W].astype(np.float32))[None].to(DEV)
        vol, _, _ = model.xsim(m, so)
        out[cond] = vol[0].cpu().numpy()
        out[cond + "_p"] = float(torch.sigmoid(model(m, so)))
    np.savez_compressed(E_PATH, name=s["name"], **out)
    print("examples ->", E_PATH)


E_PATH = HERE / "preds" / "xsim_examples.npz"

if __name__ == "__main__":
    main()
