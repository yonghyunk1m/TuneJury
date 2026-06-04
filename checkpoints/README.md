# Pretrained checkpoints

The TuneJury head checkpoints (~11 MB each) are included directly in this repository — no separate download. They are also mirrored on the Hugging Face Hub at [`TuneJury/tunejury`](https://huggingface.co/TuneJury/tunejury), so you can pull one directly: `Scorer.from_pretrained(hf_hub_download("TuneJury/tunejury", "tunejury.pt"))`.

## Primary checkpoint

`tunejury.pt`: the released bench-clean checkpoint that backs all numbers in Sections 4-5 and the appendix.

## Auxiliary design-space checkpoints

Abbreviations: **MA** = Music Arena, **MP** = MusicPrefs, **AIME** (no shorter form), **SE** = SongEval.

| File                              | Training mix                       | Used by                                   |
|-----------------------------------|------------------------------------|-------------------------------------------|
| `tunejury_leave_MA.pt`              | MP + AIME + SE                     | Leave-MA-out (fair-eval on bench MA)      |
| `tunejury_leave_MP.pt`              | MA + AIME + SE                     | Leave-MP-out                              |
| `tunejury_leave_AIME.pt`            | MA + MP + SE                       | Leave-AIME-out                            |
| `tunejury_leave_SE.pt`              | MA + MP + AIME                     | Leave-SE-out                              |
| `tunejury_leave_SE_MA.pt`           | MP + AIME                          | Double-leave-(SE+MA)-out                  |
| `tunejury_leave_MP_MA.pt`           | AIME + SE                          | Double-leave-(MP+MA)-out                  |
| `tunejury_muq_leave_MA.pt`    | MP + AIME + SE on MuQ-MuLan-large  | Encoder-swap probe (Appendix D)         |
| `tunejury_mp_zero.pt`         | MA + MP + AIME + SE (MusicPrefs text branch zero-vectored at training) | MusicPrefs prompt-blind ablation (Section 3) |

Each backs the corresponding row in paper Table 4 (`tab:head_to_head`) and the training-mix ablation (`tab:training_mix` in Appendix D). The `tunejury_mp_zero.pt` variant trains with the same 4-dataset mix as the released checkpoint, but feeds a 512-d zero vector instead of the CLAP text embedding for MusicPrefs pairs (since MusicPrefs annotators rated without prompt access); reproducible via `python -m tunejury.train --musicprefs-zero-vector ...`.

## Commercial-friendly variant (Apache-2.0)

`input_ablation/A1_clap_audio_only.pt`: a CLAP-audio-only head (512-d input, Row A1 in the input ablation, Appendix C; $0.705$ overall, tied with the seed-matched A7 retrain). Because it uses only the Apache-2.0 LAION-CLAP-Music encoder (no MERT), it is released under **Apache-2.0** for commercial use, unlike the CC-BY-NC 4.0 heads above. Also mirrored on the Hub as [`A1_clap_audio_only.pt`](https://huggingface.co/TuneJury/tunejury). It is a 512-d head, so load it with the matching `input_ablation/A1_clap_audio_only.json` config (not the default 2048-d `Scorer`).

## Frozen encoder checkpoints (downloaded automatically)

`Scorer.from_pretrained` downloads the LAION-CLAP-Music checkpoint on first use (~2.2 GB) and loads MERT-v1-330M through `transformers.AutoModel.from_pretrained("m-a-p/MERT-v1-330M")`. The LAION-CLAP-Music checkpoint is not redistributed here.

To pre-download manually:

```
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt \\
     -O music_audioset_epoch_15_esc_90.14.pt
```

## License

All TuneJury head checkpoints are released under CC-BY-NC 4.0, tracking the MERT-v1-330M upstream license, **except `input_ablation/A1_clap_audio_only.pt`** (CLAP-only, no MERT) which is **Apache-2.0** for commercial use. See `../LICENSE`.
