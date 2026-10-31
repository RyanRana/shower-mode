"""Sing-along guard: keeps AirPods Conversation Awareness ON, except while you are
singing along to the song that is playing.

Singing = your words match the synced lyrics, or your pitch follows the song's melody
(melody.py; covers songs without lyrics). Any other voice on the mic (speaker.py)
always lets Conversation Awareness through.

Fail-safe: any doubt, error, pause, or exit turns Conversation Awareness back on.
If the AirPods toggle helper does not work, falls back to ducking system volume
when you talk over music or someone else talks.
"""
from __future__ import annotations

import atexit
import collections
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

from melody import SingClassifier, SystemAudio
from speaker import SpeakerCheck

HERE = Path(__file__).resolve().parent
HELPER = HERE / "airpods_ca"

SAMPLE_RATE = 16_000
SNIPPET_SEC = 1.5
STEP_SEC = 0.75
LYRIC_WINDOW_SEC = 6.0       # lyric lines considered around the playback position
MATCH_SCORE = 70             # rapidfuzz partial_ratio threshold
MATCHES_TO_SUPPRESS = 2      # consecutive matches before turning CA off
MISMATCH_RESTORE_SEC = 2.0   # talking (not lyrics) this long -> CA back on
MAX_OFF_SEC = 60.0           # CA off without a fresh match this long -> back on
DUCK_FACTOR = 0.3
SING_MODEL = HERE / "sing_model.joblib"  # exported by tune/climb.py --export
SPEAKER_MODEL = HERE / "models" / "wespeaker_en_voxceleb_resnet34.onnx"
SPEAKER_SEC = 3.0            # 1.5 s is too short to tell voices apart reliably (tune/tune.py)
MIC_BUFFER_SEC = 6.0        # longest window anything reads (pitch model uses 4 s)
OTHER_IN_A_ROW = 2           # consecutive "other voice" verdicts before acting

log = logging.getLogger("encore")


# ---------- Now playing ----------

@dataclass
class Track:
    artist: str
    title: str
    position: float
    playing: bool


_PLAYER_SCRIPT = '''
if application "{app}" is running then
  tell application "{app}"
    if player state is playing then
      return (artist of current track) & "\\t" & (name of current track) & "\\t" & (player position as string)
    end if
  end tell
end if
return ""
'''


NOW_PLAYING_EVERY = 3.0  # seconds between player queries; position is extrapolated in between
_np_cache: tuple[float, Track | None] = (-1e9, None)


def now_playing() -> Track | None:
    """Cached player state: osascript costs ~50 ms and two process spawns per call."""
    global _np_cache
    t0, tr = _np_cache
    now = time.monotonic()
    if now - t0 < NOW_PLAYING_EVERY:
        return Track(tr.artist, tr.title, tr.position + (now - t0), True) if tr else None
    tr = _query_player()
    _np_cache = (now, tr)
    return tr


def _query_player() -> Track | None:
    for app in ("Spotify", "Music"):
        try:
            out = subprocess.run(["osascript", "-e", _PLAYER_SCRIPT.format(app=app)],
                                 capture_output=True, text=True, timeout=2).stdout.strip()
        except subprocess.TimeoutExpired:
            continue
        if out:
            artist, title, pos = out.split("\t")
            return Track(artist, title, float(pos.replace(",", ".")), True)
    return None


# ---------- Lyrics ----------

_LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)")
_lyrics_cache: dict[tuple[str, str], list[tuple[float, str]]] = {}


def synced_lyrics(artist: str, title: str) -> list[tuple[float, str]]:
    key = (artist, title)
    if key in _lyrics_cache:
        return _lyrics_cache[key]
    lines: list[tuple[float, str]] = []
    try:
        r = requests.get("https://lrclib.net/api/get",
                         params={"artist_name": artist, "track_name": title}, timeout=5,
                         headers={"User-Agent": "encore/1.0"})
        if r.ok and r.json().get("syncedLyrics"):
            for raw in r.json()["syncedLyrics"].splitlines():
                m = _LRC_LINE.match(raw)
                if m and m.group(3).strip():
                    lines.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    except requests.RequestException as e:
        log.warning("lyrics fetch failed: %s", e)
        return lines  # don't cache network failures
    _lyrics_cache[key] = lines
    log.info("lyrics for %s - %s: %d lines", artist, title, len(lines))
    return lines


def lyric_window(lines: list[tuple[float, str]], pos: float) -> str:
    return " ".join(t for ts, t in lines if pos - LYRIC_WINDOW_SEC <= ts <= pos + LYRIC_WINDOW_SEC)


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).strip()


# ---------- Outputs ----------

class ConversationAwareness:
    """Drives the AirPods toggle; tracks whether the helper works at all."""

    def __init__(self) -> None:
        self.off = False
        self.works = HELPER.exists() and self._run(True)
        log.info("AirPods CA control %s", "available" if self.works else "UNAVAILABLE, using volume ducking")

    def _run(self, on: bool) -> bool:
        try:
            return subprocess.run([str(HELPER), "on" if on else "off"],
                                  capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def set_off(self, off: bool) -> None:
        if off == self.off or not self.works:
            return
        if self._run(not off):
            self.off = off
            log.info("Conversation Awareness %s", "OFF (singing)" if off else "ON")
        elif not off:
            log.error("failed to re-enable Conversation Awareness")

    def restore(self) -> None:
        if self.works:
            self._run(True)
            self.off = False


class Ducker:
    def __init__(self) -> None:
        self.saved: int | None = None

    def _volume(self, v: int | None = None) -> int:
        script = f"set volume output volume {v}" if v is not None else "output volume of (get volume settings)"
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout.strip()
        return int(out) if out.isdigit() else 0

    def duck(self, on: bool) -> None:
        if on and self.saved is None:
            self.saved = self._volume()
            self._volume(int(self.saved * DUCK_FACTOR))
            log.info("ducked volume %d -> %d", self.saved, int(self.saved * DUCK_FACTOR))
        elif not on and self.saved is not None:
            self._volume(self.saved)
            log.info("restored volume %d", self.saved)
            self.saved = None


# ---------- Audio ----------

BUILTIN_MIC_NAMES = ("MacBook", "Built-in", "iMac", "Mac mini", "Mac Studio")


def mac_mic_index() -> int | None:
    """ENCORE_MIC (substring of the device name) wins, else the Mac's built-in mic."""
    wanted = (os.environ["ENCORE_MIC"],) if os.environ.get("ENCORE_MIC") else BUILTIN_MIC_NAMES
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and any(w in d["name"] for w in wanted):
            if "AirPods" not in d["name"]:
                return i
    return None  # never fall back to the AirPods mic: it forces the low-quality call codec


class Mic:
    def __init__(self) -> None:
        idx = mac_mic_index()
        if idx is None:
            raise SystemExit("built-in microphone not found; set ENCORE_MIC to part of the device name")
        self.buf: collections.deque[float] = collections.deque(maxlen=int(SAMPLE_RATE * MIC_BUFFER_SEC))
        self.lock = threading.Lock()
        self.stream = sd.InputStream(device=idx, channels=1, samplerate=SAMPLE_RATE,
                                     dtype="float32", callback=self._cb)
        self.stream.start()
        self.noise_floor = 0.003

    def _cb(self, data, frames, t, status) -> None:
        with self.lock:
            self.buf.extend(data[:, 0])

    def snippet(self, seconds: float = SNIPPET_SEC) -> np.ndarray:
        with self.lock:
            return np.array(self.buf, dtype=np.float32)[-int(SAMPLE_RATE * seconds):]

    def has_voice(self, audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if rms < self.noise_floor * 3:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms  # adapt only on quiet frames
            return False
        return True


# ---------- Main loop ----------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # two INFO lines per tick otherwise
    warnings.filterwarnings("ignore", message="invalid value encountered in cast")  # pYIN on unvoiced frames
    ca = ConversationAwareness()
    ducker = Ducker()

    def cleanup(*_):
        ca.restore()
        ducker.duck(False)
        sys.exit(0)

    atexit.register(lambda: (ca.restore(), ducker.duck(False)))
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    sysaudio = SystemAudio(HERE / "sysaudio", seconds=5)
    atexit.register(sysaudio.stop)
    try:
        tuned = HERE / "speaker_threshold.json"
        speaker = SpeakerCheck(SPEAKER_MODEL, HERE / "voice_profile.npy",
                               json.loads(tuned.read_text())["other_below"] if tuned.exists() else None)
    except Exception as e:  # missing model or sherpa failure: run without second-voice check
        log.warning("second-voice check disabled: %s", e)
        speaker = None
    # no hand-set fallback: the old melody rule called 30% of talking "singing" (tune/results.jsonl)
    sing_clf = SingClassifier(SING_MODEL) if SING_MODEL.exists() else None
    log.info("pitch check: %s", "tuned model" if sing_clf else "disabled (no sing_model.joblib)")
    mic = Mic()
    log.info("listening")

    matches = 0
    last_match = 0.0
    talking_since: float | None = None
    other_streak = 0

    while True:
        time.sleep(STEP_SEC)
        now = time.monotonic()
        track = now_playing()
        lines = synced_lyrics(track.artist, track.title) if track else []
        song = sysaudio.snippet(int(SAMPLE_RATE * max(SPEAKER_SEC, sing_clf.window if sing_clf else 0))) if sysaudio.available else np.zeros(0)
        music_on = track is not None or (song.size > 0 and float(np.sqrt(np.mean(song ** 2))) > 0.005)

        if not music_on:
            matches, talking_since = 0, None
            ca.set_off(False)
            ducker.duck(False)
            continue

        audio = mic.snippet()
        if not mic.has_voice(audio):
            talking_since = None
            ducker.duck(False)
            if ca.off and now - last_match > MAX_OFF_SEC:
                ca.set_off(False)
            continue

        emb = speaker.embed(mic.snippet(SPEAKER_SEC)) if speaker else None
        other_streak = other_streak + 1 if emb is not None and speaker.is_other(emb) else 0
        if other_streak >= OTHER_IN_A_ROW:
            # someone else is talking: always let Conversation Awareness through
            log.info("other voice (sim=%.2f)", speaker.similarity(emb))
            matches, talking_since = 0, None
            ca.set_off(False)
            ducker.duck(not ca.works)
            continue

        lyric = None
        if lines:
            segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
            heard = normalize(" ".join(s.text for s in segments))
            if len(heard) >= 4:
                lyric = fuzz.partial_ratio(heard, normalize(lyric_window(lines, track.position)))
                log.info("heard %r lyric=%d", heard, lyric)
        feats = sing_clf.features(mic.snippet(sing_clf.window), song) if sing_clf and song.size else None
        p = sing_clf.update(feats) if sing_clf else None
        melodic = p is not None and p >= sing_clf.threshold
        if p is not None:
            log.info("sing p=%.2f (threshold %.2f)", p, sing_clf.threshold)

        singing = (lyric is not None and lyric >= MATCH_SCORE) or melodic
        if lyric is None and feats is None:
            continue  # nothing to judge on; keep current state

        if singing:
            if emb is not None:
                speaker.enroll(emb)
            matches += 1
            last_match = now
            talking_since = None
            ducker.duck(False)
            if matches >= MATCHES_TO_SUPPRESS:
                ca.set_off(True)
        else:
            matches = 0
            talking_since = talking_since or now
            if now - talking_since >= MISMATCH_RESTORE_SEC:
                ca.set_off(False)
                if not ca.works:
                    ducker.duck(True)


def doctor() -> int:
    """Check every dependency once and print what works. Exit 1 if the service can't run."""
    def row(ok: bool | None, name: str, detail: str) -> None:
        print(f"  {'ok ' if ok else 'off' if ok is None else 'BAD'}  {name:<22} {detail}")

    print("encore doctor")
    idx = mac_mic_index()
    row(idx is not None, "microphone", sd.query_devices(idx)["name"] if idx is not None else "not found (set ENCORE_MIC)")

    tr = _query_player()
    row(True if tr else None, "now playing", f"{tr.artist} - {tr.title}" if tr else "nothing playing in Spotify/Music")

    sysaudio = HERE / "sysaudio"
    if sysaudio.exists():
        p = subprocess.Popen([str(sysaudio)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = p.communicate(timeout=3)
            row(False, "system audio", err.decode(errors="replace").strip()[:90] or "helper exited")
        except subprocess.TimeoutExpired:
            p.terminate()
            row(True, "system audio", "capturing")
    else:
        row(False, "system audio", "helper not built; run ./install.sh")

    r = subprocess.run([str(HELPER), "on"], capture_output=True, text=True, timeout=5) if HELPER.exists() else None
    code = r.returncode if r else -1
    msg = {0: "toggle works", 2: "no AirPods connected", 3: "macOS blocks the AACP channel"}.get(
        code, r.stderr.strip() if r else "helper not built")
    row(True if code == 0 else None, "AirPods CA toggle", msg if code == 0 else f"{msg}; volume ducking instead")

    row(SING_MODEL.exists(), "pitch model", SING_MODEL.name if SING_MODEL.exists() else "missing")
    row(SPEAKER_MODEL.exists(), "speaker model", SPEAKER_MODEL.name if SPEAKER_MODEL.exists() else "missing; run ./install.sh")
    profile = HERE / "voice_profile.npy"
    n = len(np.load(profile)) if profile.exists() else 0
    row(True if n >= 5 else None, "voice profile", f"{n} samples (second-voice check needs 5)")
    return 0 if idx is not None else 1


if __name__ == "__main__":
    if "--doctor" in sys.argv[1:]:
        sys.exit(doctor())
    main()
