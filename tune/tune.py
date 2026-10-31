"""Offline tuning for the second-voice check, plus shared data helpers for climb.py.
Touches no audio devices.

Speaker check (LibriSpeech dev-clean, laptop-mic simulation): 5-clip enrollment centroid vs
held-out clips of the same and other speakers, scored the way the service decides
(3 s windows, "other" only after two verdicts in a row). Writes ../speaker_threshold.json.
The sing-along classifier is tuned by climb.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from speaker import SpeakerCheck  # noqa: E402

SR = 16_000
WIN = int(1.5 * SR)
MAX_FP = 0.02
SPK_SEC = 3.0
SPK_MAX_FA = 0.05
rng = np.random.default_rng(0)
_LP = butter(4, 6000, fs=SR, output="sos")


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    return librosa.resample(x.mean(axis=1) if x.ndim == 2 else x, orig_sr=sr, target_sr=SR).astype(np.float32)


def as_mic(x: np.ndarray, leak_from: np.ndarray | None = None, detune: bool = True) -> np.ndarray:
    """Laptop-mic simulation: detune, lowpass, gain, noise, a little song bleed.

    detune=False for speaker tuning: phase-vocoder artifacts make all voices embed alike.
    """
    if detune and rng.random() < 0.5:
        x = librosa.effects.pitch_shift(x, sr=SR, n_steps=rng.uniform(-0.3, 0.3))
    x = sosfilt(_LP, x).astype(np.float32)
    x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.05, 0.4)
    if leak_from is not None:
        x = x + leak_from * rng.uniform(0.0, 0.05)
    # noise relative to the signal: the live service only acts on audio well above the room floor
    snr_db = rng.uniform(15, 30)
    noise_rms = np.sqrt(np.mean(x ** 2)) / 10 ** (snr_db / 20)
    return (x + rng.normal(0, noise_rms, x.size)).astype(np.float32)


def load_speech(root: Path) -> dict[str, list[np.ndarray]]:
    by_spk: dict[str, list[np.ndarray]] = {}
    for f in sorted(root.rglob("*.flac")):
        by_spk.setdefault(f.parts[-3], []).append(f)
    out = {}
    for spk, files in by_spk.items():
        clips = []
        for f in files[:12]:
            y, sr = sf.read(f, dtype="float32")
            clips.append(y if sr == SR else to16k(y, sr))
        out[spk] = clips
    return out


def tune_speaker(speech: dict[str, list[np.ndarray]]) -> dict:
    sc = SpeakerCheck(ROOT / "models" / "wespeaker_en_voxceleb_resnet34.onnx", Path("/dev/null/none"))
    # mirrors the service: 3 s windows, "other" only after 2 verdicts in a row (score = max of a pair)
    w = int(SPK_SEC * SR)
    emb = {spk: [sc.embed(as_mic(c[s:s + w], detune=False)) for c in clips for s in range(0, c.size - w, w)][:24]
           for spk, clips in speech.items()}
    same, other = [], []
    spks = [s for s in emb if len(emb[s]) >= 10]
    pairs = lambda v: [max(a, b) for a, b in zip(v[::2], v[1::2])]  # noqa: E731
    for spk in spks:
        e = emb[spk]
        cen = np.mean(e[:5], axis=0)
        cen /= np.linalg.norm(cen)
        same += pairs([float(x @ cen) for x in e[5:]])
        for o in spks:
            if o != spk:
                other += pairs([float(x @ cen) for x in emb[o][5:9]])
    same, other = np.array(same), np.array(other)
    # a conversation produces a new pair every 0.75 s, so 5% per pair still catches it within seconds
    thr = float(np.quantile(other, 1 - SPK_MAX_FA))
    print(f"speaker check: {len(spks)} speakers  threshold={thr:.3f}  "
          f"others accepted as you={(other >= thr).mean():.1%}  you flagged as other={(same < thr).mean():.1%}")
    return {"other_below": thr, "false_accept": float((other >= thr).mean()),
            "false_reject": float((same < thr).mean())}


def main() -> None:
    data = ROOT / "data"
    print("loading speech...")
    speech = load_speech(data / "libri" / "LibriSpeech" / "dev-clean")
    (ROOT / "speaker_threshold.json").write_text(json.dumps(tune_speaker(speech), indent=2))


if __name__ == "__main__":
    main()
