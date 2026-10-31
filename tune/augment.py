"""Realistic sing-along augmentation and the "realistic" eval set. Offline only.

The clean eval (climb.py) uses the studio vocal stem through tune.as_mic: perfectly in tune,
in time, with weak dry bleed. Real singing along is not like that. Every *voice* condition
here (sung, other_song, shifted, talk) goes through the same pipeline so processing
artifacts are never a label cue:

  - timing: global lag 0-400 ms plus a slow random warp (+-0..80 ms, periods 3-12 s)
  - pitch: global detune, per-note error (segments 0.15-0.6 s, sd 10-40 cents),
    occasional octave shift of a phrase (out-of-range singers)
  - timbre: cepstral formant shift (x0.85-1.18) and a random smooth EQ
  - dropped / skipped phrases (0.4-2 s gaps)

and every condition (bleed-only included) gets the same stronger bleed: the song through a
speaker-like highpass, a synthetic room impulse response (RT60 0.15-0.8 s), 0-40 ms delay,
at -24..-3 dB relative to the voice.

New hard negative "shifted": your own song's vocal from a different part of the song
(>5 s away, and not a repeat of the same melody, see SHIFT_MAX_SIM).

Usage: python augment.py --decode     decode MUSDB18 to a 16 kHz memmap (data/xsim_audio.npy, ~2.3 GB, temporary)
       python augment.py --build      build data/realistic_signals.npz (same chunks as climb's full set)
       python augment.py --speech     LibriSpeech pool as data/xsim_speech_{audio,offs}.npy (memmapped by xsim.py workers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

SR = 16_000
HOP = 320
AUDIO = ROOT / "data" / "xsim_audio.npy"           # int16 (2, total): row 0 mix, row 1 vocals
INDEX = ROOT / "data" / "xsim_index.npz"           # offsets, lengths, vocal rms + chroma frames
REAL = ROOT / "data" / "realistic_signals.npz"     # realistic eval set (all splits)
CHUNKS, CHUNK_SEC = 3, 15
SHIFT_MIN = 5.0
SHIFT_MAX_SIM = 0.75   # reject shifted vocals whose chroma matches the true vocal this well (repeated chorus)
CONDS_REAL = ("sung", "talk", "other_song", "bleed", "shifted")

_LP6 = butter(4, 6000, fs=SR, output="sos")
_LP36 = butter(6, 3600, fs=SR, output="sos")
NFFT, SHOP = 1024, 256


# ---------- voice augmentation ----------

def _box(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average (same length), via cumsum."""
    c = np.concatenate([[0.0], np.cumsum(np.pad(x.astype(np.float64), (k // 2, k - 1 - k // 2), mode="edge"))])
    return (c[k:] - c[:-k]) / k


def _smooth_noise(n: int, rng, periods=(3.0, 12.0), k: int = 3) -> np.ndarray:
    """Sum of k random sinusoids, unit peak, on a grid of n points at 100 Hz."""
    t = np.arange(n) / 100.0
    out = np.zeros(n)
    for _ in range(k):
        out += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * t / rng.uniform(*periods) + rng.uniform(0, 2 * np.pi))
    return out / (np.abs(out).max() + 1e-9)


def _cents_track(n_src: int, rng, octave_p: float = 0.15) -> np.ndarray:
    """Pitch error in cents per source sample: detune + per-note error + optional octave phrase."""
    seg_ends, pos = [], 0
    while pos < n_src:
        pos += int(rng.uniform(0.15, 0.6) * SR)
        seg_ends.append(min(pos, n_src))
    sd = rng.uniform(10, 40)
    vals = np.clip(rng.normal(0, sd, len(seg_ends)), -70, 70)
    c = np.repeat(vals, np.diff([0] + seg_ends)) + rng.normal(0, 15)
    if rng.random() < octave_p:
        frac = rng.uniform(0.3, 1.0)
        L = int(frac * n_src)
        s0 = int(rng.integers(0, n_src - L + 1))
        c[s0:s0 + L] += -1200 if rng.random() < 0.7 else 1200
    k = int(0.03 * SR)  # 30 ms glides between notes
    c = _box(c, k)
    return c


def _phase_vocoder(Z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Sample STFT Z (F, M) at fractional frame positions pos (N,), vectorized phase accumulation."""
    M = Z.shape[1]
    pos = np.clip(pos, 0, M - 2.0001)
    k = np.floor(pos).astype(int)
    a = pos - k
    A, P = np.abs(Z), np.angle(Z)
    mag = (1 - a) * A[:, k] + a * A[:, k + 1]
    adv = np.linspace(0, np.pi * SHOP, Z.shape[0])[:, None]
    dphi = P[:, k + 1] - P[:, k] - adv
    dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi)) + adv
    phase = P[:, k[:1]] + np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(dphi[:, :-1], axis=1)], 1)
    return mag * np.exp(1j * phase)


def _formant_eq(Y: np.ndarray, rng, alpha: float, lifter: int = 24) -> np.ndarray:
    """Shift the spectral envelope by alpha (cepstral smoothing) and apply a random smooth EQ."""
    F = Y.shape[0]
    logm = np.log(np.abs(Y) + 1e-7)
    cep = np.fft.irfft(logm, axis=0)
    cep[lifter:-lifter] = 0
    env = np.fft.rfft(cep, axis=0).real[:F]
    src = np.clip(np.arange(F) / alpha, 0, F - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, F - 1)
    w = (src - i0)[:, None]
    env2 = (1 - w) * env[i0] + w * env[i1]
    f = np.linspace(0, SR / 2, F)
    lf = np.log2(np.maximum(f, 50) / 50)
    eq_db = rng.uniform(-3, 3) * (lf / lf.max() - 0.5)  # tilt
    for _ in range(int(rng.integers(1, 4))):
        eq_db += rng.uniform(-6, 6) * np.exp(-0.5 * ((lf - rng.uniform(0.5, 7)) / rng.uniform(0.3, 1.0)) ** 2)
    return Y * np.exp(env2 - env + (eq_db / 20 * np.log(10))[:, None])


def augment_voice(src: np.ndarray, a: int, L: int, rng, drops: bool = True,
                  max_lag: float = 0.4, warp_amp: float = 0.08) -> np.ndarray:
    """Augmented voice for output samples [a, a+L) of the (song-aligned) source track src.

    Output time t reads source time t - d(t): d = lag 0..max_lag + slow warp. Pitch error is
    attached to source time (i.e. to notes). Pitch and timing are decoupled: pitch by
    time-varying resampling, timing by a phase vocoder sampling the resampled STFT.
    """
    margin = int(1.0 * SR)
    s0 = a - margin
    seg = np.zeros(L + 2 * margin, np.float32)
    lo, hi = max(0, s0), min(src.size, a + L + margin)
    if hi > lo:
        seg[lo - s0:hi - s0] = src[lo:hi]
    if not np.any(seg):
        return np.zeros(L, np.float32)
    cents = _cents_track(seg.size, rng)
    if np.any(cents > 600):
        seg = sosfilt(_LP36, seg).astype(np.float32)
    r = 2.0 ** (cents / 1200)
    u_of_s = np.concatenate([[0.0], np.cumsum(1.0 / r[:-1])])         # z-time (samples) per source sample
    z = np.interp(np.arange(0, u_of_s[-1]), u_of_s, seg).astype(np.float32)  # pitch shifted by r
    Z = librosa.stft(z, n_fft=NFFT, hop_length=SHOP, center=True)
    n_out = L // SHOP + 2
    t = np.arange(n_out) * SHOP                                      # output sample (relative to a)
    d100 = rng.uniform(0, max_lag) + rng.uniform(0, warp_amp) * _smooth_noise(L // 160 + 2, rng)
    d = np.interp(t / 160, np.arange(d100.size), d100) * SR
    s_src = t - d + margin                                            # position in seg
    u = np.interp(s_src, np.arange(seg.size), u_of_s)
    Y = _phase_vocoder(Z, u / SHOP)
    Y = _formant_eq(Y, rng, alpha=float(np.exp(rng.uniform(np.log(0.85), np.log(1.18)))))
    y = librosa.istft(Y, hop_length=SHOP, n_fft=NFFT, length=L).astype(np.float32)
    # keep the source level so the "vocals present" rule (rms >= 0.02) means the same thing
    ref = seg[margin:margin + L]
    ry, rr = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(ref ** 2))
    y *= rr / (ry + 1e-9)
    if drops:
        env = np.ones(L, np.float32)
        for _ in range(rng.poisson(L / SR / 8)):
            dur = int(rng.uniform(0.4, 2.0) * SR)
            st = int(rng.integers(0, max(1, L - dur)))
            env[st:st + dur] = 0
        k = int(0.05 * SR)
        env = _box(env, k).astype(np.float32)
        y *= env
    return y


# ---------- bleed + mic ----------

def room_leak(song: np.ndarray, L: int, rng) -> np.ndarray:
    """Last L samples of the song as heard by the mic from the speakers: highpass, room, delay."""
    hp = butter(2, rng.uniform(100, 400), btype="high", fs=SR, output="sos")
    x = sosfilt(hp, song).astype(np.float32)
    rt60 = rng.uniform(0.15, 0.8)
    n = int(rt60 * 1.2 * SR)
    tt = np.arange(n) / SR
    h = rng.normal(0, 1, n) * np.exp(-6.9 * tt / rt60)
    h[: int(0.003 * SR)] = 0
    drr_db = rng.uniform(-6, 10)
    h *= 10 ** (-drr_db / 20) / (np.sqrt(np.sum(h ** 2)) + 1e-9)
    h[0] = 1.0
    h = np.concatenate([np.zeros(int(rng.uniform(0, 0.04) * SR)), h])
    y = fftconvolve(x, h)[: x.size]
    return y[-L:].astype(np.float32)


def mic(voice: np.ndarray | None, leak: np.ndarray, rng, bleed: str = "strong",
        song: np.ndarray | None = None) -> np.ndarray:
    """Mic signal. bleed='strong': room leak at -24..-3 dB re voice (bleed-only: leak at voice level).
    bleed='weak': the clean as_mic recipe (dry song * U(0, 0.05), no room), for the ablation."""
    if bleed == "weak":
        song = leak if song is None else song
        if voice is None:
            x = sosfilt(_LP6, song).astype(np.float32)
            x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.05, 0.4) + song * rng.uniform(0, 0.05)
        else:
            x = sosfilt(_LP6, voice).astype(np.float32)
            x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.05, 0.4) + song * rng.uniform(0, 0.05)
    else:
        lk = sosfilt(_LP6, leak).astype(np.float32)
        if voice is None or not np.any(voice):
            x = lk / (np.abs(lk).max() + 1e-9) * rng.uniform(0.05, 0.4)
        else:
            v = sosfilt(_LP6, voice).astype(np.float32)
            v = v / (np.abs(v).max() + 1e-9) * rng.uniform(0.05, 0.4)
            ratio = 10 ** (rng.uniform(-24, -3) / 20)
            x = v + lk * ratio * np.sqrt(np.mean(v ** 2)) / (np.sqrt(np.mean(lk ** 2)) + 1e-9)
    snr_db = rng.uniform(15, 30)
    noise_rms = np.sqrt(np.mean(x ** 2)) / 10 ** (snr_db / 20)
    return (x + rng.normal(0, noise_rms, x.size)).astype(np.float32)


# ---------- shifted-vocal selection ----------

def chroma_sim(ca: np.ndarray, cb: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> float:
    """Best (over +-500 ms lags) mean cosine of two vocal chroma sequences (12, n), on frames
    where both are voiced."""
    best = 0.0
    n = ca.shape[1]
    for lag in range(-25, 26, 3):
        i = np.arange(max(0, -lag), min(n, n - lag))
        m = (ra[i] > 0.02) & (rb[i + lag] > 0.02)
        if m.sum() < 25:
            continue
        x, y = ca[:, i[m]], cb[:, i[m] + lag]
        cs = (x * y).sum(0) / (np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0) + 1e-9)
        best = max(best, float(cs.mean()))
    return best


def pick_shift(a: int, L: int, n: int, chroma: np.ndarray, rms: np.ndarray, rng, tries: int = 12) -> int | None:
    """Start b of a same-song vocal stretch >5 s from a, voiced, and not the same melody."""
    fa, fl = a // HOP, L // HOP
    for _ in range(tries):
        b = int(rng.integers(0, max(1, n - L)))
        if abs(b - a) < SHIFT_MIN * SR:
            continue
        fb = b // HOP
        if np.mean(rms[fb:fb + fl] > 0.02) < 0.3:
            continue
        if chroma_sim(chroma[:, fa:fa + fl], chroma[:, fb:fb + fl], rms[fa:fa + fl], rms[fb:fb + fl]) < SHIFT_MAX_SIM:
            return b
    return None


# ---------- decoded audio ----------

def decode() -> None:
    import musdb
    from joblib import Parallel, delayed
    import tune as T
    db = musdb.DB(root=str(ROOT / "data" / "musdb18"), download=False)
    est = [int(t.duration * SR) + SR for t in db]
    offs = np.concatenate([[0], np.cumsum(est)]).astype(np.int64)
    mm = np.lib.format.open_memmap(AUDIO, mode="w+", dtype=np.int16, shape=(2, int(offs[-1])))
    del mm

    def one(i):
        t = musdb.DB(root=str(ROOT / "data" / "musdb18"), download=False)[i]
        mix, voc = T.to16k(t.audio, t.rate), T.to16k(t.targets["vocals"].audio, t.rate)
        n = min(mix.size, voc.size, est[i])
        m = np.load(AUDIO, mmap_mode="r+")
        m[0, offs[i]:offs[i] + n] = np.clip(mix[:n] * 32767, -32768, 32767).astype(np.int16)
        m[1, offs[i]:offs[i] + n] = np.clip(voc[:n] * 32767, -32768, 32767).astype(np.int16)
        m.flush()
        rms = librosa.feature.rms(y=voc[:n], frame_length=1024, hop_length=HOP)[0]
        ch = librosa.feature.chroma_stft(y=voc[:n], sr=SR, n_fft=2048, hop_length=HOP)
        return n, t.name, rms.astype(np.float32), ch.astype(np.float16)

    out = Parallel(n_jobs=6, verbose=5)(delayed(one)(i) for i in range(len(db)))
    lens = np.array([o[0] for o in out])
    fo = np.concatenate([[0], np.cumsum([o[2].size for o in out])]).astype(np.int64)
    np.savez(INDEX, offs=offs, lens=lens, names=np.array([o[1] for o in out]), frame_offs=fo,
             rms=np.concatenate([o[2] for o in out]), chroma=np.concatenate([o[3] for o in out], axis=1))
    print(f"decoded {len(db)} songs, {lens.sum() / SR / 3600:.2f} h -> {AUDIO}")


_CACHE: dict = {}


def songs() -> "Songs":
    if "songs" not in _CACHE:
        _CACHE["songs"] = Songs()
    return _CACHE["songs"]


def speech_pool() -> list[np.ndarray]:
    if "speech" not in _CACHE:
        _CACHE["speech"] = load_speech_pool()
    return _CACHE["speech"]


class Songs:
    """Read access to the decoded songs (float32 views on demand)."""

    def __init__(self) -> None:
        self.audio = np.load(AUDIO, mmap_mode="r")
        d = np.load(INDEX)
        self.offs, self.lens, self.names = d["offs"], d["lens"], d["names"]
        self.fo, self._rms, self._chroma = d["frame_offs"], d["rms"], d["chroma"]

    def __len__(self) -> int:
        return len(self.lens)

    def mix(self, i):
        return self.audio[0, self.offs[i]:self.offs[i] + self.lens[i]].astype(np.float32) / 32767

    def voc(self, i):
        return self.audio[1, self.offs[i]:self.offs[i] + self.lens[i]].astype(np.float32) / 32767

    def rms(self, i):
        return self._rms[self.fo[i]:self.fo[i + 1]]

    def chroma(self, i):
        return self._chroma[:, self.fo[i]:self.fo[i + 1]].astype(np.float32)


def chunk_starts(voc: np.ndarray) -> np.ndarray:
    """The same 3 most-vocal 15 s stretches climb.build_full_signals uses."""
    L = CHUNK_SEC * SR
    starts = np.arange(0, voc.size - L, L)
    active = [np.mean(librosa.feature.rms(y=voc[a:a + L], hop_length=SR)[0] > 0.02) for a in starts]
    return starts[np.argsort(active)[::-1][:CHUNKS]]


SPEECH = ROOT / "data" / "xsim_speech"


def save_speech() -> None:
    """LibriSpeech dev-clean pool (same clips as climb) as one int16 array, for training workers."""
    clips = load_speech_pool()
    np.save(f"{SPEECH}_audio.npy", np.concatenate([np.clip(c * 32767, -32768, 32767).astype(np.int16) for c in clips]))
    np.save(f"{SPEECH}_offs.npy", np.concatenate([[0], np.cumsum([c.size for c in clips])]))


def load_speech_pool() -> list[np.ndarray]:
    import tune as T
    return [c for cl in T.load_speech(ROOT / "data" / "libri" / "LibriSpeech" / "dev-clean").values() for c in cl]


def talk_source(speech, L: int, rng) -> np.ndarray:
    parts, n = [], 0
    while n < L + 2 * SR:
        c = speech[int(rng.integers(len(speech)))]
        parts.append(c)
        n += c.size
    return np.concatenate(parts).astype(np.float32)


def build_realistic(seed: int = 0) -> None:
    """Realistic eval/train signals on the same chunks as climb's full clean set."""
    import climb
    from joblib import Parallel, delayed
    S = Songs()
    parts = climb.split(len(S))
    by_split = {p: [i for i in range(len(S)) if parts[i] == p] for p in ("train", "dev", "holdout")}
    L = CHUNK_SEC * SR

    def one(i):
        S, speech = songs(), speech_pool()
        rng = np.random.default_rng([seed, i])
        mix_full, voc_full = S.mix(i), S.voc(i)
        ch, rms = S.chroma(i), S.rms(i)
        rows = []
        for ci, a in enumerate(chunk_starts(voc_full)):
            a = int(a)
            m = mix_full[a:a + L]
            pre = mix_full[max(0, a - SR):a + L]
            voc = augment_voice(voc_full, a, L, rng)
            pool = [j for j in by_split[parts[i]] if j != i]
            j = pool[int(rng.integers(len(pool)))]
            vj = S.voc(j)
            cj = chunk_starts(vj)
            b = int(cj[int(rng.integers(len(cj)))]) if len(cj) else 0
            other = augment_voice(vj, b, L, rng)
            bs = pick_shift(a, L, voc_full.size, ch, rms, rng, tries=40)
            shifted = augment_voice(voc_full, bs, L, rng) if bs is not None else np.zeros(L, np.float32)
            tk = talk_source(speech, L, rng)
            talk = augment_voice(tk, SR, L, rng, drops=False)
            leak = lambda: room_leak(pre, L, rng)  # noqa: E731  fresh room/level per condition
            e = {"name": str(S.names[i]), "song": i, "chunk": ci, "start": a, "shift_ok": bs is not None,
                 "mix": m, "voc": voc,
                 "sung": mic(voc, leak(), rng), "talk": mic(talk, leak(), rng),
                 "other_song": mic(other, leak(), rng), "bleed": mic(None, leak(), rng),
                 "shifted": mic(shifted, leak(), rng)}
            rows.append({k: v.astype(np.float16) if isinstance(v, np.ndarray) else v for k, v in e.items()})
        return rows

    del S
    out = Parallel(n_jobs=8, verbose=5)(delayed(one)(i) for i in range(150))
    sig = {n: r for n, r in enumerate(r for rs in out for r in rs)}
    np.savez(REAL, signals=np.array(sig, dtype=object))
    print(f"realistic set: {len(sig)} chunks, shifted ok in {np.mean([s['shift_ok'] for s in sig.values()]):.0%}"
          f" -> {REAL}")


def load_realistic() -> dict:
    return np.load(REAL, allow_pickle=True)["signals"].item()


if __name__ == "__main__":
    if "--decode" in sys.argv:
        decode()
    if "--speech" in sys.argv:
        save_speech()
    if "--build" in sys.argv:
        build_realistic()
