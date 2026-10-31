"""Pitch check: does the voice on the mic follow the melody of what's playing?

Song audio comes from the sysaudio helper (system output capture). The voice's pitch
track (YIN) is compared to the song's chroma at the same moments: when singing along,
voiced frames land on pitch classes the song is sounding; speech wanders off them.
Random speech scores around 0.25-0.35 (3 of 12 pitch classes), singing along 0.6+.
"""
from __future__ import annotations

import collections
import logging
import subprocess
import threading
import time
from pathlib import Path

import joblib
import librosa
import numpy as np

SAMPLE_RATE = 16_000
HOP = 320  # 20 ms frames
TOP_CLASSES = 3
MAX_LAG_FRAMES = 15  # +-300 ms voice/song alignment slack

log = logging.getLogger("encore")


class SystemAudio:
    """Ring buffer fed by the sysaudio helper.

    The helper dies when ScreenCaptureKit loses its display (screen lock, sleep, monitor
    unplugged) or permission is missing. It is restarted with exponential backoff, so
    the pitch check comes back on its own. .available is False while capture is down.
    """

    RESTART_MIN_SEC = 2.0
    RESTART_MAX_SEC = 60.0
    HEALTHY_SEC = 30.0  # a run this long resets the backoff

    def __init__(self, helper: Path, seconds: float) -> None:
        self.helper = helper
        self.buf: collections.deque[float] = collections.deque(maxlen=int(SAMPLE_RATE * seconds))
        self.lock = threading.Lock()
        self.received = 0
        self.proc: subprocess.Popen | None = None
        self._stopped = threading.Event()
        if helper.exists():
            threading.Thread(target=self._supervise, daemon=True).start()
        else:
            log.warning("system audio helper missing (%s); run ./install.sh", helper)

    def _supervise(self) -> None:
        delay = self.RESTART_MIN_SEC
        while not self._stopped.is_set():
            started = time.monotonic()
            self.received = 0
            self.proc = subprocess.Popen([str(self.helper)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            err = self._pump(self.proc)
            if self._stopped.is_set():
                return
            with self.lock:
                self.buf.clear()  # stale song audio must not be compared to a live voice
            if time.monotonic() - started >= self.HEALTHY_SEC:
                delay = self.RESTART_MIN_SEC
            if "-3801" in err:
                log.warning("system audio: no Screen & System Audio Recording permission; pitch check off")
            else:
                log.warning("system audio capture ended (%s); retrying in %.0f s", err or "no output", delay)
            if self._stopped.wait(delay):
                return
            delay = min(delay * 2, self.RESTART_MAX_SEC)

    def _pump(self, proc: subprocess.Popen) -> str:
        assert proc.stdout and proc.stderr
        while chunk := proc.stdout.read(3200):
            samples = np.frombuffer(chunk[: len(chunk) // 4 * 4], dtype=np.float32)
            with self.lock:
                if self.received == 0:
                    log.info("system audio capture running")
                self.buf.extend(samples)
                self.received += samples.size
        proc.wait()
        return proc.stderr.read().decode(errors="replace").strip()

    @property
    def available(self) -> bool:
        return self.received > 0 and self.proc is not None and self.proc.poll() is None

    def snippet(self, n: int) -> np.ndarray:
        with self.lock:
            return np.array(list(self.buf)[-n:], dtype=np.float32)

    def stop(self) -> None:
        self._stopped.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def voice_pitch_classes(voice: np.ndarray) -> np.ndarray:
    """Pitch class (0-11) per frame, -1 where unvoiced."""
    f0 = librosa.yin(voice, fmin=80, fmax=1000, sr=SAMPLE_RATE, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=voice, frame_length=1024, hop_length=HOP)[0][: f0.size]
    voiced = rms > max(0.01, np.percentile(rms, 40))
    pcs = np.full(f0.size, -1)
    pcs[voiced] = np.round(librosa.hz_to_midi(f0[voiced])).astype(int) % 12
    return pcs


def melody_score(voice: np.ndarray, song: np.ndarray) -> float | None:
    """Fraction of voiced frames on one of the song's strongest pitch classes; None if not enough signal."""
    if voice.size < SAMPLE_RATE or song.size < voice.size or np.sqrt(np.mean(song ** 2)) < 1e-3:
        return None
    pcs = voice_pitch_classes(voice)
    chroma = librosa.feature.chroma_stft(y=song[-voice.size:], sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP)
    top = np.argsort(chroma, axis=0)[-TOP_CLASSES:]  # (TOP, frames)
    n = min(pcs.size, top.shape[1])
    voiced_idx = np.flatnonzero(pcs[:n] >= 0)
    if voiced_idx.size < 15:  # < 300 ms of voiced sound
        return None
    best = 0.0
    for lag in range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1, 3):
        idx = voiced_idx + lag
        ok = (idx >= 0) & (idx < n)
        if ok.sum() < 10:
            continue
        hits = (top[:, idx[ok]] == pcs[voiced_idx[ok]]).any(axis=0)
        best = max(best, float(hits.mean()))
    return best


def pitch_stability(voice: np.ndarray) -> float:
    """Share of voiced frames that hold the same pitch class as the previous frame; singing holds notes."""
    pcs = voice_pitch_classes(voice)
    v = pcs >= 0
    pairs = v[1:] & v[:-1]
    return float((pcs[1:][pairs] == pcs[:-1][pairs]).mean()) if pairs.sum() >= 10 else 0.0


def f0_track(x: np.ndarray, tracker: str) -> np.ndarray:
    if tracker == "pyin":
        f0, voiced, _ = librosa.pyin(x, fmin=80, fmax=1000, sr=SAMPLE_RATE, frame_length=1024, hop_length=HOP)
        return np.where(voiced, f0, np.nan)
    f0 = librosa.yin(x, fmin=80, fmax=1000, sr=SAMPLE_RATE, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=HOP)[0][: f0.size]
    return np.where(rms > max(0.01, np.percentile(rms, 40)), f0, np.nan)


def window_features(voice: np.ndarray, song: np.ndarray, tracker: str = "pyin",
                    contour: bool = False) -> np.ndarray | None:
    """Feature vector for the sing-along classifier (see tune/climb.py); None if too little voice."""
    f0 = f0_track(voice, tracker)
    voiced = ~np.isnan(f0)
    if voiced.sum() < 15:
        return None
    midi = librosa.hz_to_midi(f0[voiced])
    pcs = np.full(f0.size, -1)
    pcs[voiced] = np.round(midi).astype(int) % 12

    chroma = librosa.feature.chroma_stft(y=song, sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP)
    n = min(pcs.size, chroma.shape[1])
    top = np.argsort(chroma[:, :n], axis=0)[-3:]
    vidx = np.flatnonzero(pcs[:n] >= 0)
    best_mel, best_lag = 0.0, 0
    for lag in range(-15, 16, 3):
        idx = vidx + lag
        ok = (idx >= 0) & (idx < n)
        if ok.sum() >= 10:
            m = float((top[:, idx[ok]] == pcs[vidx[ok]]).any(axis=0).mean())
            if m > best_mel:
                best_mel, best_lag = m, lag

    pairs = (pcs[1:] >= 0) & (pcs[:-1] >= 0)
    stability = float((pcs[1:][pairs] == pcs[:-1][pairs]).mean()) if pairs.sum() >= 10 else 0.0
    offset = np.angle(np.mean(np.exp(2j * np.pi * midi))) / (2 * np.pi)
    quantized = float((np.abs(((midi - offset) + 0.5) % 1 - 0.5) < 0.2).mean())
    pitch_range = float(np.percentile(midi, 90) - np.percentile(midi, 10))
    nfft = 1 << (2 * voice.size - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(voice, nfft) * np.conj(np.fft.rfft(song, nfft)), nfft)
    leak = float(np.abs(np.concatenate([xc[:1600], xc[-1600:]])).max()
                 / (np.linalg.norm(voice) * np.linalg.norm(song) + 1e-9))
    feats = [best_mel, stability, float(voiced.mean()), quantized, pitch_range, leak]

    if contour:
        # does the voice's pitch move when and where the song's dominant pitch class moves?
        song_pc = np.argmax(chroma[:, :n], axis=0)
        sh = np.roll(song_pc, -best_lag)
        both = (pcs[:n] >= 0)
        both[1:] &= both[:-1]
        both[0] = False
        if both.sum() >= 10:
            v_move = pcs[1:n] != pcs[:n - 1]
            s_move = sh[1:n] != sh[:n - 1]
            b = both[1:n]
            agree = float((v_move[b] == s_move[b]).mean())
            # circular semitone distance to the song's dominant class at aligned frames
            d = np.abs((pcs[:n][both] - sh[both] + 6) % 12 - 6)
            feats += [agree, float(d.mean()), float((d == 0).mean())]
        else:
            feats += [0.5, 3.0, 0.0]
    return np.array(feats, dtype=np.float64)



# ---------- v2 features (tune/rounds.py) ----------
# `e` needs .groups, .lag, .lag_step, .extra; `w` holds per-frame tracks for one window (see frames()).

def _hits(top, pcs, vidx, lag, n):
    idx = vidx + lag
    ok = (idx >= 0) & (idx < n)
    if ok.sum() < 10:
        return None
    return float((top[:, idx[ok]] == pcs[vidx[ok]]).any(axis=0).mean())


def _soft(ch, pcs, vidx, lag, n):
    """Mean normalized song chroma energy at the voice's pitch class (soft hit rate)."""
    idx = vidx + lag
    ok = (idx >= 0) & (idx < n)
    if ok.sum() < 10:
        return None
    c = ch[:, idx[ok]]
    c = c / (c.max(axis=0, keepdims=True) + 1e-6)
    return float(c[pcs[vidx[ok]], np.arange(ok.sum())].mean())


def frame_features(w: dict, e) -> np.ndarray | None:
    f0, st, cq = w["f0"], w["song_st"], w["song_cq"]
    voiced = ~np.isnan(f0)
    if voiced.sum() < 15:
        return None
    midi = 69 + 12 * np.log2(f0[voiced] / 440.0)
    pcs = np.full(f0.size, -1)
    pcs[voiced] = np.round(midi).astype(int) % 12
    n = min(pcs.size, st.shape[1])
    vidx = np.flatnonzero(pcs[:n] >= 0)
    top = np.argsort(st[:, :n], axis=0)[-3:]
    lags = range(-e.lag, e.lag + 1, e.lag_step)
    mels = {lag: _hits(top, pcs, vidx, lag, n) for lag in lags}
    mels = {k: v for k, v in mels.items() if v is not None}
    best_lag = max(mels, key=mels.get) if mels else 0
    best_mel = mels.get(best_lag, 0.0)

    out = []
    if "base" in e.groups:
        pairs = (pcs[1:] >= 0) & (pcs[:-1] >= 0)
        stability = float((pcs[1:][pairs] == pcs[:-1][pairs]).mean()) if pairs.sum() >= 10 else 0.0
        offset = np.angle(np.mean(np.exp(2j * np.pi * midi))) / (2 * np.pi)
        quantized = float((np.abs(((midi - offset) + 0.5) % 1 - 0.5) < 0.2).mean())
        prange = float(np.percentile(midi, 90) - np.percentile(midi, 10))
        out += [best_mel, stability, float(voiced.mean()), quantized, prange, w["leak"]]
    if "contrast" in e.groups:
        # key-level agreement (any lag up to +-2.4 s) vs. the best local alignment: singing along
        # matches *here*, a different part of the song or another song in the same key matches anywhere
        far = [v for lag in range(-120, 121, 6) if abs(lag) > 40
               for v in [_hits(top, pcs, vidx, lag, n)] if v is not None]
        null = float(np.median(far)) if far else best_mel
        out += [best_mel - null, null, float(np.max(far)) if far else best_mel]
    if "soft" in e.groups:
        s = [_soft(cq, pcs, vidx, lag, n) for lag in lags]
        s = [v for v in s if v is not None]
        out += [max(s) if s else 0.0, _soft(cq, pcs, vidx, best_lag, n) or 0.0]
    if "vchroma" in e.groups:
        # mic CQT chroma vs song chroma, cosine per frame, over voiced frames, best lag; contrasted
        mc = w["mic_cq"][:, :n].astype(np.float32)
        sc = cq[:, :n].astype(np.float32)
        mc = mc / (np.linalg.norm(mc, axis=0, keepdims=True) + 1e-6)
        sc = sc / (np.linalg.norm(sc, axis=0, keepdims=True) + 1e-6)

        def cos(lag):
            idx = vidx + lag
            ok = (idx >= 0) & (idx < n)
            return float((mc[:, vidx[ok]] * sc[:, idx[ok]]).sum(0).mean()) if ok.sum() >= 10 else None
        near = [v for lag in lags for v in [cos(lag)] if v is not None]
        far = [v for lag in range(-120, 121, 8) if abs(lag) > 40 for v in [cos(lag)] if v is not None]
        b = max(near) if near else 0.0
        out += [b, b - (float(np.median(far)) if far else b)]
    if "onset" in e.groups:
        mo, so = w["mic_on"].astype(np.float32), w["song_on"].astype(np.float32)
        m = min(mo.size, so.size)
        mo, so = mo[:m] - mo[:m].mean(), so[:m] - so[:m].mean()
        den = np.linalg.norm(mo) * np.linalg.norm(so) + 1e-6
        xc = [float(np.dot(mo[max(0, -l):m - max(0, l)], so[max(0, l):m - max(0, -l)]) / den)
              for l in range(-25, 26)]
        out += [max(xc), float(np.argmax(xc) - 25) / 25]
    if "lagstats" in e.groups:
        v = np.array(list(mels.values())) if mels else np.zeros(1)
        out += ([] if e.extra.get("no_lagpos") else [float(best_lag) / max(e.lag, 1)]) + [float(v.mean()), float(np.sort(v)[-2] if v.size > 1 else v[0])]
    if "vp" in e.groups:
        vp = w["vp"].astype(np.float32)
        out += [float(vp.mean()), float(np.percentile(vp, 90)), float(np.median(w["rms"].astype(np.float32)))]
    return np.array(out, dtype=np.float64)



def frames(voice: np.ndarray, song: np.ndarray) -> dict:
    """Per-frame tracks for frame_features, computed the same way as tune/rawcache.py."""
    f0, _, vp = librosa.pyin(voice, fmin=80, fmax=1000, sr=SAMPLE_RATE, frame_length=1024, hop_length=HOP)
    return {"f0": f0, "vp": vp, "rms": librosa.feature.rms(y=voice, frame_length=1024, hop_length=HOP)[0],
            "mic_cq": librosa.feature.chroma_cqt(y=voice, sr=SAMPLE_RATE, hop_length=HOP),
            "mic_on": librosa.onset.onset_strength(y=voice, sr=SAMPLE_RATE, hop_length=HOP),
            "song_st": librosa.feature.chroma_stft(y=song, sr=SAMPLE_RATE, n_fft=2048, hop_length=HOP),
            "song_cq": librosa.feature.chroma_cqt(y=song, sr=SAMPLE_RATE, hop_length=HOP),
            "song_on": librosa.onset.onset_strength(y=song, sr=SAMPLE_RATE, hop_length=HOP),
            "leak": leak(voice, song)}


def leak(voice: np.ndarray, song: np.ndarray) -> float:
    """Peak raw cross-correlation within +-100 ms: how much of the song leaks straight into the mic."""
    nfft = 1 << (2 * voice.size - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(voice, nfft) * np.conj(np.fft.rfft(song, nfft)), nfft)
    return float(np.abs(np.concatenate([xc[:1600], xc[-1600:]])).max()
                 / (np.linalg.norm(voice) * np.linalg.norm(song) + 1e-9))


class SingClassifier:
    """Model exported by tune/climb.py (v1) or tune/rounds.py --export (v2): features over `window`
    seconds, probability averaged over the last `smooth` decisions, singing at `threshold`."""

    def __init__(self, path: Path) -> None:
        b = joblib.load(path)
        self.model, self.threshold = b["model"], b["threshold"]
        self.window = b["window"]
        self.v2 = b.get("version") == 2
        if self.v2:
            from types import SimpleNamespace
            self.exp = SimpleNamespace(groups=tuple(b["groups"]), lag=b["lag"], lag_step=b["lag_step"],
                                       extra=b["extra"])
        else:
            self.tracker, self.contour = b["tracker"], b["contour"]
        self.recent: collections.deque[float] = collections.deque(maxlen=b["smooth"])

    def features(self, voice: np.ndarray, song: np.ndarray) -> np.ndarray | None:
        n = int(self.window * SAMPLE_RATE)
        if voice.size < n or song.size < n or np.sqrt(np.mean(song[-n:] ** 2)) < 1e-3:
            return None
        if self.v2:
            return frame_features(frames(voice[-n:], song[-n:]), self.exp)
        return window_features(voice[-n:], song[-n:], self.tracker, self.contour)

    def update(self, x: np.ndarray | None) -> float | None:
        """Feed one decision step; returns the smoothed probability (None resets the history)."""
        if x is None:
            self.recent.clear()
            return None
        self.recent.append(float(self.model.predict_proba(x[None])[0, 1]))
        return float(np.mean(self.recent))
