"""Public scoring API for TuneJury.

Loads the released checkpoint and the frozen LAION-CLAP-Music +
MERT-v1-330M encoders, and exposes a single ``score(audio, prompt)``
entry point that returns a scalar reward.

Example
-------
>>> from tunejury import Scorer
>>> scorer = Scorer.from_pretrained(
...     "checkpoints/tunejury.pt"
... )
>>> reward = scorer.score("clip.wav", prompt="a calm piano piece")

The released checkpoint uses an empty prompt by default on
out-of-distribution prompt formats (Section 4.2 of the paper). Pass
``prompt=""`` to reproduce the OOD recommendation, or pass an
arena-style request to use the text branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torchaudio

from .model import TuneJury

try:
    import laion_clap  # type: ignore
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "TuneJury inference requires laion_clap and transformers. "
        "Install via `pip install -r requirements.txt`."
    ) from exc


CLAP_CKPT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)
CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
MERT_HF_NAME = "m-a-p/MERT-v1-330M"
MERT_SR = 24_000
CLAP_SR = 48_000


@dataclass
class Scorer:
    """End-to-end TuneJury scoring pipeline."""

    head: TuneJury
    clap_model: object  # laion_clap module
    mert_processor: object
    mert_model: object
    device: str

    @classmethod
    def from_pretrained(
        cls,
        head_ckpt_path: str,
        clap_ckpt_path: str | None = None,
        device: str | None = None,
    ) -> "Scorer":
        """Load TuneJury head and the frozen encoders."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        head = TuneJury(input_dim=2048)
        state = torch.load(head_ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
        head.eval().to(device)

        clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        clap_ckpt_path = clap_ckpt_path or _resolve_clap_ckpt(head_ckpt_path)
        clap_model.load_ckpt(ckpt=clap_ckpt_path)
        clap_model.to(device).eval()

        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_HF_NAME, trust_remote_code=True
        )
        mert_model = AutoModel.from_pretrained(
            MERT_HF_NAME, trust_remote_code=True
        ).to(device).eval()

        return cls(
            head=head,
            clap_model=clap_model,
            mert_processor=mert_processor,
            mert_model=mert_model,
            device=device,
        )

    @torch.no_grad()
    def score(self, audio_path: str, prompt: str = "") -> float:
        """Score a single audio clip against an (optional) prompt."""
        features = self._extract_features(audio_path, prompt)
        score = self.head(features).item()
        return score

    @torch.no_grad()
    def score_batch(
        self, audio_paths: Sequence[str], prompts: Sequence[str] | None = None,
    ) -> list[float]:
        prompts = prompts or [""] * len(audio_paths)
        return [
            self.score(p, prompt=prompt)
            for p, prompt in zip(audio_paths, prompts)
        ]

    @torch.no_grad()
    def score_waveform(
        self, wav, sr: int, text: str | None = None,
    ) -> float:
        """Score a raw waveform (numpy or torch) without writing to disk.

        Used by eval/sanity.py to score synthetic adversarial inputs
        (silence, noise, sines) without round-tripping through a file.
        Empty/None ``text`` follows the paper §3 convention: 512-d zero
        vector at the text branch.
        """
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(np.asarray(wav))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
            wav = wav.t()
        wav = wav.float()
        wav = wav.mean(dim=0, keepdim=True)  # mono

        # CLAP audio: resample to 48 kHz
        wav_clap = wav
        if sr != CLAP_SR:
            wav_clap = torchaudio.functional.resample(wav, sr, CLAP_SR)
        clap_audio_emb = self.clap_model.get_audio_embedding_from_data(
            x=wav_clap.cpu().numpy(), use_tensor=False
        )
        clap_audio = torch.tensor(clap_audio_emb[0], dtype=torch.float32).flatten()

        # MERT audio: resample to 24 kHz
        wav_mert = wav
        if sr != MERT_SR:
            wav_mert = torchaudio.functional.resample(wav, sr, MERT_SR)
        mert_inputs = self.mert_processor(
            wav_mert.squeeze(0).cpu().numpy(),
            sampling_rate=MERT_SR,
            return_tensors="pt",
        ).to(self.device)
        mert_out = self.mert_model(**mert_inputs, output_hidden_states=True)
        mert_audio = (
            mert_out.last_hidden_state.mean(dim=1).squeeze(0).flatten().cpu().float()
        )

        clap_text = self._clap_text_embed(text or "")
        features = torch.cat(
            [clap_audio, mert_audio, clap_text], dim=-1
        ).unsqueeze(0).to(self.device)
        return float(self.head(features).item())

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio_path: str, prompt: str) -> torch.Tensor:
        clap_audio = self._clap_audio_embed(audio_path)         # (512,)
        clap_text = self._clap_text_embed(prompt)               # (512,)
        mert_audio = self._mert_audio_embed(audio_path)         # (1024,)
        # Training-time concatenation order: [clap_audio, mert_audio, clap_text].
        features = torch.cat(
            [clap_audio, mert_audio, clap_text], dim=-1
        ).unsqueeze(0)  # (1, 2048)
        return features.to(self.device)

    def _clap_audio_embed(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != CLAP_SR:
            wav = torchaudio.functional.resample(wav, sr, CLAP_SR)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        emb = self.clap_model.get_audio_embedding_from_data(
            x=wav.cpu().numpy(), use_tensor=False
        )
        return torch.tensor(emb[0], dtype=torch.float32).flatten()

    def _clap_text_embed(self, text: str) -> torch.Tensor:
        # Training convention (paper §3): empty prompt → 512-d zero vector,
        # not the CLAP encoding of "". Matches feature-extraction pipeline.
        if not text:
            return torch.zeros(512, dtype=torch.float32)
        emb = self.clap_model.get_text_embedding([text], use_tensor=False)
        return torch.tensor(emb[0], dtype=torch.float32).flatten()

    def _mert_audio_embed(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != MERT_SR:
            wav = torchaudio.functional.resample(wav, sr, MERT_SR)
        wav = wav.mean(dim=0)  # mono
        inputs = self.mert_processor(
            wav.cpu().numpy(),
            sampling_rate=MERT_SR,
            return_tensors="pt",
        ).to(self.device)
        out = self.mert_model(**inputs, output_hidden_states=True)
        # Canonical training-time pooling: last layer, mean over time.
        # Matches the feature-extraction pipeline used to build the
        # TuneJury training corpus and the DifferentiableScorer code path.
        return out.last_hidden_state.mean(dim=1).squeeze(0).flatten().cpu().float()


def _resolve_clap_ckpt(head_ckpt_path: str) -> str:
    """Locate the LAION-CLAP-Music checkpoint, downloading if missing."""
    ckpt_dir = os.path.dirname(head_ckpt_path)
    path = os.path.join(ckpt_dir, CLAP_CKPT_NAME)
    if os.path.exists(path):
        return path
    print(f"[TuneJury] LAION-CLAP checkpoint not found at {path}, downloading...")
    torch.hub.download_url_to_file(CLAP_CKPT_URL, path)
    return path
