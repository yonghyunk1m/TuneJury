"""Extract LAION-CLAP-Music + MERT-v1-330M features for every paired clip.

Paper anchor: Section 3 (Inputs paragraph) + §A.D (Encoder swap probe).

For each pair, the released TuneJury head consumes a 2,048-d concatenation
of [CLAP audio (512), MERT audio (1024), CLAP text (512)]. SongEval pairs
have no source prompt; their CLAP text branch receives a 512-d zero
vector at training (Section 3).

This script writes one ``.pt`` per pair under ``--out-dir``, keyed by
the pair's ``battle_uuid`` or pair identifier. Resumable: existing
output files are skipped.

Each output ``.pt`` blob carries the schema consumed by
``tunejury.dataset.TuneJuryDataset``:

  uuid     : str
  prompt   : str (may be "")
  text_emb : 512-d CLAP text embedding (zeros for empty prompt)
  clap_a   : 512-d CLAP audio embedding of clip A
  mert_a   : 1024-d MERT audio embedding of clip A
  clap_b   : 512-d CLAP audio embedding of clip B
  mert_b   : 1024-d MERT audio embedding of clip B
  flag     : 1-d float tensor source flag
             (1.0 for prompt-conditioned sources {Music Arena, AIME,
             SongEval}, 0.0 for MusicPrefs; matches the released
             ``data/processed_features/*.pt`` blobs).
  winner   : "model_a" | "model_b" | "tie"
  source   : "MusicArena" | "MusicPrefs" | "AIME" | "SongEval"

For the MuQ-MuLan encoder swap variant (Appendix `tab:encoder_swap`), pass
``--encoders muq``; the head trains on the joint 1,024-d MuQ-MuLan
output instead of the 2,048-d CLAP+MERT stack.

Usage
-----
$ python data/extract_features.py \\
      --pairs       data/splits/all_pairs.json \\
      --audio-root  /path/to/audio \\
      --out-dir     data/processed_features \\
      --encoders    clap mert        # default: CLAP audio+text + MERT audio
$ python data/extract_features.py \\
      --pairs       data/splits/all_pairs.json \\
      --audio-root  /path/to/audio \\
      --out-dir     data/muq_features \\
      --encoders    muq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio


def _load_clap(device: str):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    ckpt = Path("checkpoints/music_audioset_epoch_15_esc_90.14.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"LAION-CLAP-Music ckpt expected at {ckpt}")
    model.load_ckpt(str(ckpt))
    model.eval()
    return model


def _load_mert(device: str):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    name = "m-a-p/MERT-v1-330M"
    feat = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    return feat, model


def _load_muq(device: str):
    from muq import MuQMuLan

    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()
    return model


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return _resample(wav, sr, target_sr).squeeze(0)


def _clap_audio_emb(clap, wav: torch.Tensor, device: str) -> torch.Tensor:
    wav_48k = _resample(wav.unsqueeze(0), 16000, 48000).squeeze(0)
    with torch.no_grad():
        emb = clap.get_audio_embedding_from_data(
            x=wav_48k.unsqueeze(0).to(device), use_tensor=True
        )
    return emb.squeeze(0).cpu()  # 512-d


def _clap_text_emb(clap, text: str | None, device: str) -> torch.Tensor:
    if text is None or text == "":
        return torch.zeros(512)
    with torch.no_grad():
        emb = clap.get_text_embedding([text], use_tensor=True)
    return emb.squeeze(0).cpu()  # 512-d


def _mert_audio_emb(feat, mert, wav: torch.Tensor, device: str) -> torch.Tensor:
    inputs = feat(wav.numpy(), sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mert(**inputs, output_hidden_states=False)
    # time-mean over the final layer (Section 3 Inputs paragraph)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # 1024-d


def _muq_emb(muq, wav: torch.Tensor, text: str | None, device: str):
    with torch.no_grad():
        out = muq(
            wavs=wav.unsqueeze(0).to(device),
            texts=[text or ""],
        )
    return out.audio_embeds.squeeze(0).cpu(), out.text_embeds.squeeze(0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--encoders",
        nargs="+",
        choices=["clap", "mert", "muq"],
        default=["clap", "mert"],
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.pairs.open() as fp:
        pairs = json.load(fp)

    use_clap = "clap" in args.encoders
    use_mert = "mert" in args.encoders
    use_muq = "muq" in args.encoders
    if use_muq and (use_clap or use_mert):
        raise ValueError("MuQ swap mode is exclusive (--encoders muq only).")

    clap = _load_clap(device) if use_clap else None
    mert_feat = mert = None
    if use_mert:
        mert_feat, mert = _load_mert(device)
    muq = _load_muq(device) if use_muq else None

    # Released blobs use flag=1.0 for prompt-conditioned sources
    # {MusicArena, AIME, SongEval} and flag=0.0 for MusicPrefs. The
    # source string is preserved verbatim so downstream per-dataset
    # eval scripts (eval/internal_per_dataset.py, eval/aime_per_axis.py)
    # group correctly.
    _SOURCE_FLAG = {
        "MusicArena": 1.0,
        "AIME": 1.0,
        "SongEval": 1.0,
        "MusicPrefs": 0.0,
    }
    _WINNER_NORMALIZE = {
        "model_a": "model_a",
        "model_b": "model_b",
        "a": "model_a",
        "b": "model_b",
        "tie": "tie",
        "draw": "tie",
        "": "tie",
    }

    for pair in pairs:
        pair_id = pair.get("pair_id") or pair.get("battle_uuid") or pair.get("uuid")
        if pair_id is None:
            raise KeyError(
                "pair record must carry one of 'pair_id', 'battle_uuid', 'uuid'"
            )
        out_path = args.out_dir / f"{pair_id}.pt"
        if out_path.exists():
            continue

        # Normalize the winner label up-front so TuneJuryDataset's
        # WINNER_TO_TARGET lookup (model_a -> 1.0, model_b -> 0.0,
        # tie -> 0.5) succeeds.
        raw_winner = str(pair.get("winner", "tie")).strip().lower()
        winner = _WINNER_NORMALIZE.get(raw_winner, raw_winner)
        if winner not in ("model_a", "model_b", "tie"):
            raise ValueError(
                f"pair {pair_id}: unrecognized winner label "
                f"{pair.get('winner')!r}; expected one of "
                "{'model_a', 'model_b', 'tie'} (or 'a'/'b'/'draw')."
            )

        source = pair.get("source") or pair.get("dataset") or ""
        if source not in _SOURCE_FLAG:
            raise ValueError(
                f"pair {pair_id}: source {source!r} not in "
                f"{sorted(_SOURCE_FLAG)}; per-dataset eval would skip it."
            )
        flag_value = _SOURCE_FLAG[source]

        prompt = pair.get("prompt", "") or ""

        blob: dict = {
            "uuid": str(pair_id),
            "prompt": prompt,
            "flag": torch.tensor([flag_value], dtype=torch.float32),
            "winner": winner,
            "source": source,
        }

        for side in ("a", "b"):
            audio_path = args.audio_root / pair[f"{side}_audio"]
            wav_16k = _load_audio(audio_path, target_sr=16000)
            wav_24k = _load_audio(audio_path, target_sr=24000) if use_mert else None
            side_text = pair.get(f"{side}_prompt") or prompt

            if use_clap:
                blob[f"clap_{side}"] = _clap_audio_emb(clap, wav_16k, device)
            if use_mert:
                blob[f"mert_{side}"] = _mert_audio_emb(mert_feat, mert, wav_24k, device)
            if use_muq:
                a_emb, t_emb = _muq_emb(muq, wav_16k, side_text, device)
                blob[f"muq_audio_{side}"] = a_emb
                if side == "a":
                    blob["muq_text"] = t_emb

        if use_clap:
            # Empty prompts (e.g. SongEval) get a 512-d zero vector,
            # matching the released blobs and Section 3 of the paper.
            blob["text_emb"] = _clap_text_emb(clap, prompt, device)

        torch.save(blob, out_path)

    print(f"wrote {len(pairs)} feature blobs to {args.out_dir}")


if __name__ == "__main__":
    main()
