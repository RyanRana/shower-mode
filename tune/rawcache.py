"""Per-frame raw tracks for the realistic set, so new window features can be tried in seconds.

For every chunk (15 s) and condition: pYIN f0 + voiced probability, frame RMS, mic chroma (CQT),
onset envelope; for the song mix: STFT chroma, CQT chroma, onset envelope. All at 20 ms hop over
the whole chunk, so windows are frame slices. The mic/song leak (raw cross-correlation) needs
the waveform, so it is computed here per window for each window length in WINDOWS.

Output: data/raw_realistic.npz. Offline only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from augment import CONDS_REAL, load_realistic  # noqa: E402

SR, HOP = 16_000, 320
STEP = int(0.75 * SR)
WINDOWS = (3.0, 4.0)
OUT = ROOT / "data" / "raw_realistic.npz"


def leak(voice: np.ndarray, song: np.ndarray) -> float:
    nfft = 1 << (2 * voice.size - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(voice, nfft) * np.conj(np.fft.rfft(song, nfft)), nfft)
    return float(np.abs(np.concatenate([xc[:1600], xc[-1600:]])).max()
                 / (np.linalg.norm(voice) * np.linalg.norm(song) + 1e-9))


def tracks(x: np.ndarray) -> dict:
    f0, _, vp = librosa.pyin(x, fmin=80, fmax=1000, sr=SR, frame_length=1024, hop_length=HOP)
    return {"f0": f0.astype(np.float32), "vp": vp.astype(np.float16),
            "rms": librosa.feature.rms(y=x, frame_length=1024, hop_length=HOP)[0].astype(np.float16),
            "cq": librosa.feature.chroma_cqt(y=x, sr=SR, hop_length=HOP).astype(np.float16),
            "on": librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP).astype(np.float16)}


def one(i: int, s: dict) -> dict:
    s = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in s.items()}
    mix = s["mix"]
    out = {"song": s["song"], "chunk": s["chunk"], "shift_ok": s.get("shift_ok", True),
           "song_st": librosa.feature.chroma_stft(y=mix, sr=SR, n_fft=2048, hop_length=HOP).astype(np.float16),
           "song_cq": librosa.feature.chroma_cqt(y=mix, sr=SR, hop_length=HOP).astype(np.float16),
           "song_on": librosa.onset.onset_strength(y=mix, sr=SR, hop_length=HOP).astype(np.float16),
           "voc_rms": np.array([np.sqrt(np.mean(s["voc"][a:a + int(w * SR)] ** 2))
                                for w in WINDOWS for a in range(0, mix.size - int(w * SR) + 1, STEP)]),
           "conds": {}}
    for cond in CONDS_REAL:
        x = s[cond]
        d = tracks(x)
        d["leak"] = {w: np.array([leak(x[a:a + int(w * SR)], mix[a:a + int(w * SR)])
                                  for a in range(0, mix.size - int(w * SR) + 1, STEP)]) for w in WINDOWS}
        out["conds"][cond] = d
    return out


def main() -> None:
    sig = load_realistic()
    keys = sorted(sig)
    res = Parallel(n_jobs=-1, verbose=5)(delayed(one)(i, sig[i]) for i in keys)
    del sig
    np.savez(OUT, raw=np.array(dict(zip(keys, res)), dtype=object))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
