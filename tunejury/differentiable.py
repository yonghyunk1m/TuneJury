"""Differentiable TuneJury scoring pipeline (for Mode 2 DITTO).

The default :class:`tunejury.Scorer` loads audio from disk and is not
differentiable end-to-end. ``DifferentiableScorer`` instead accepts a
waveform tensor and lets gradients flow through MERT-v1-330M back to
the input. LAION-CLAP audio is detached for numpy interop, so only
the MERT path carries gradient. The TuneJury MLP head and the MERT
body are kept on-graph but with all parameters frozen -- only the
input waveform is updated by DITTO.

Example
-------
>>> import torch
>>> from tunejury.differentiable import DifferentiableScorer
>>> scorer = DifferentiableScorer(
...     reward_ckpt="checkpoints/tunejury.pt",
... )
>>> wav = torch.randn(1, 16000 * 10, requires_grad=True)  # 10 s @ 16 kHz
>>> reward = scorer.score_waveform(wav, sr=16000, text="instrumental")
>>> (-reward).backward()
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchaudio

from .model import TuneJury


CLAP_CKPT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)
CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
MERT_HF_NAME = "m-a-p/MERT-v1-330M"


class DifferentiableScorer(nn.Module):
    """Differentiable TuneJury scoring pipeline for DITTO-style loops."""

    def __init__(
        self,
        reward_ckpt: str,
        clap_ckpt: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device

        # TuneJury MLP head (frozen weights, gradients flow through).
        self.head = TuneJury(input_dim=2048)
        state = torch.load(reward_ckpt, map_location="cpu", weights_only=False)
        self.head.load_state_dict(state if "state_dict" not in state else state["state_dict"])
        self.head.eval()

        # LAION-CLAP-Music (CLAP audio is detached for numpy interop).
        import laion_clap  # type: ignore
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        clap_ckpt = clap_ckpt or _resolve_clap_ckpt(reward_ckpt)
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(
            *a, **{**kw, "weights_only": kw.get("weights_only", False)}
        )
        try:
            self.clap.load_ckpt(ckpt=clap_ckpt)
        finally:
            torch.load = _orig
        self.clap.eval()

        # MERT-v1-330M (frozen weights, gradients flow through).
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_HF_NAME, trust_remote_code=True,
        )
        self.mert = AutoModel.from_pretrained(MERT_HF_NAME, trust_remote_code=True)
        self.mert.eval()

        # Freeze everything; gradients only flow w.r.t. the input.
        for p in self.parameters():
            p.requires_grad = False

        self.to(device)
        self._text_emb_cache: dict[str, torch.Tensor] = {}

    def get_text_embedding(self, text: str) -> torch.Tensor:
        # Training convention (paper §3): empty prompt → 512-d zero vector,
        # not CLAP encoding of "". Matches feature-extraction pipeline.
        if not text:
            if "__zeros__" not in self._text_emb_cache:
                self._text_emb_cache["__zeros__"] = torch.zeros(
                    512, device=self.device, dtype=torch.float32,
                )
            return self._text_emb_cache["__zeros__"]
        if text not in self._text_emb_cache:
            with torch.no_grad():
                emb = self.clap.get_text_embedding([text], use_tensor=True)[0]
            self._text_emb_cache[text] = emb.to(self.device)
        return self._text_emb_cache[text]

    def score_waveform(
        self, waveform: torch.Tensor, sr: int = 16_000, text: str = "",
    ) -> torch.Tensor:
        """Score a (T,) or (B, T) waveform. Gradients flow through CLAP + MERT
        back to the input waveform (DITTO-compatible).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        wav_48k = torchaudio.functional.resample(waveform, sr, 48_000)
        wav_24k = torchaudio.functional.resample(waveform, sr, 24_000)

        # CLAP audio embedding via tensor path (use_tensor=True preserves grad).
        clap_audio = self.clap.get_audio_embedding_from_data(
            x=wav_48k.to(self.device), use_tensor=True,
        )
        clap_audio = clap_audio[0].float()

        # MERT audio embedding via tensor path (skip Wav2Vec2FeatureExtractor,
        # which does cpu().numpy() conversion. MERT's processor only adds a
        # batch dim and attention mask of all ones at do_normalize=False, so
        # we build them directly to preserve the graph).
        wav_24k_input = wav_24k.to(self.device).float()
        attn_mask = torch.ones_like(wav_24k_input, dtype=torch.long)
        mert_out = self.mert(
            input_values=wav_24k_input,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        mert_emb = mert_out.last_hidden_state.mean(dim=1).squeeze(0).float()

        # CLAP text embedding (cached, no grad needed w.r.t. waveform).
        text_emb = self.get_text_embedding(text)

        # Concatenated 2048-d input, scored by the TuneJury head.
        # Training-time order: [clap_audio, mert_audio, text_emb].
        feat = torch.cat([clap_audio, mert_emb, text_emb]).unsqueeze(0)
        return self.head(feat).squeeze()


def _resolve_clap_ckpt(reward_ckpt: str) -> str:
    ckpt_dir = os.path.dirname(reward_ckpt)
    path = os.path.join(ckpt_dir, CLAP_CKPT_NAME)
    if os.path.exists(path):
        return path
    print(f"[TuneJury] LAION-CLAP checkpoint missing at {path}, downloading ...")
    torch.hub.download_url_to_file(CLAP_CKPT_URL, path)
    return path
