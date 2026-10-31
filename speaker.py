"""Second-voice detection: is the voice on the mic you, or someone else?

Self-enrolling: every snippet confidently identified as you singing along (lyrics or
melody matched) is added to your voice profile, so it needs no setup. Until the
profile has MIN_ENROLL samples, it reports "unknown" and never blocks anything.
The profile persists in voice_profile.npy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16_000
MIN_ENROLL = 5
MAX_PROFILE = 50
OTHER_BELOW = 0.45  # cosine similarity to your profile below this = someone else

log = logging.getLogger("shower-mode")


class SpeakerCheck:
    def __init__(self, model: Path, profile: Path, other_below: float | None = None) -> None:
        self.profile_path = profile
        self.other_below = OTHER_BELOW if other_below is None else other_below
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2))
        self.samples: list[np.ndarray] = list(np.load(profile)) if profile.exists() else []
        log.info("voice profile: %d samples%s", len(self.samples),
                 "" if self.enrolled else f" (second-voice check starts at {MIN_ENROLL})")

    @property
    def enrolled(self) -> bool:
        return len(self.samples) >= MIN_ENROLL

    def embed(self, audio: np.ndarray) -> np.ndarray:
        # fixed peak level: the embedding model separates speakers best around 0.1 (see tune/tune.py)
        audio = (audio / (np.abs(audio).max() + 1e-9) * 0.1).astype(np.float32)
        s = self.extractor.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio)
        s.input_finished()
        e = np.array(self.extractor.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) or 1.0)

    def similarity(self, emb: np.ndarray) -> float | None:
        if not self.enrolled:
            return None
        centroid = np.mean(self.samples, axis=0)
        return float(emb @ (centroid / np.linalg.norm(centroid)))

    def is_other(self, emb: np.ndarray) -> bool:
        sim = self.similarity(emb)
        return sim is not None and sim < self.other_below

    def enroll(self, emb: np.ndarray) -> None:
        self.samples = (self.samples + [emb])[-MAX_PROFILE:]
        np.save(self.profile_path, np.stack(self.samples))
        if len(self.samples) == MIN_ENROLL:
            log.info("voice profile ready; second-voice check active")
