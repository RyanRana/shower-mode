"""Learned sing-along detector: no hand-made features.

Frozen MERT-v1-95M (self-supervised music model) encodes the raw mic window and the raw
song window; a small transformer head attends across both streams and outputs
P(the mic voice is singing along to this song). Trained by tune/e2e.py.
MERT weights are CC-BY-NC 4.0 (personal use).
"""
from __future__ import annotations

import librosa
import numpy as np
import torch
from torch import nn

MERT_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24_000
LAYERS = range(4, 9)  # mid layers carry pitch/harmony information
POOL = 3              # 75 Hz -> 25 Hz frames
D = 128


class MertEncoder:
    def __init__(self, device: str | None = None) -> None:
        from transformers import AutoModel
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True).eval().to(self.device)
        self._outs: list[torch.Tensor] = []
        for i in LAYERS:  # this remote code ignores output_hidden_states, so hook the layers
            self.model.encoder.layers[i].register_forward_hook(
                lambda m, a, o: self._outs.append(o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def __call__(self, windows: list[np.ndarray], sr: int = 16_000) -> np.ndarray:
        """(N, frames, 768) float16 for N equal-length windows."""
        x = np.stack([librosa.resample(w, orig_sr=sr, target_sr=MERT_SR) for w in windows])
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)
        self._outs.clear()
        self.model(torch.from_numpy(x).float().to(self.device))
        h = torch.stack(self._outs).mean(0)
        h = nn.functional.avg_pool1d(h.transpose(1, 2), POOL).transpose(1, 2)
        return h.half().cpu().numpy()


class SingHead(nn.Module):
    """Joint attention over [CLS, voice frames, song frames] -> singing-along logit."""

    def __init__(self, frames: int, layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.voice_in = nn.Sequential(nn.Linear(768, D), nn.LayerNorm(D))
        self.song_in = nn.Sequential(nn.Linear(768, D), nn.LayerNorm(D))
        self.pos = nn.Parameter(torch.zeros(1, frames, D))
        self.kind = nn.Parameter(torch.zeros(2, D))
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.kind, std=0.02)
        block = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.out = nn.Linear(D, 1)

    def forward(self, voice: torch.Tensor, song: torch.Tensor) -> torch.Tensor:
        n = voice.shape[0]
        v = self.voice_in(voice) + self.pos + self.kind[0]
        s = self.song_in(song) + self.pos + self.kind[1]
        h = self.encoder(torch.cat([self.cls.expand(n, -1, -1), v, s], dim=1))
        return self.out(h[:, 0]).squeeze(-1)
