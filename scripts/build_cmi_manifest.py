"""Build cmi_rewardbench manifest with proper a/b mixed winners from all_test.jsonl.

The CMI-RewardBench public release's manifest.json normalizes all pairs so
'winner' is always 'b' (audio_b is always preferred). Our eval/cmi_rewardbench.py
expects a/b mixed winners (the original preference field from all_test.jsonl).

Usage:
    python scripts/build_cmi_manifest.py \
        --bench-root /path/to/CMI-RewardBench/data \
        --out /path/to/CMI-RewardBench/data/manifest_proper.json

Then pass --bench-root pointing to the directory containing manifest_proper.json
(rename or symlink to manifest.json).
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", required=True, type=Path,
                        help="CMI-RewardBench data root containing all_test.jsonl")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output manifest.json path")
    args = parser.parse_args()

    all_rows = [json.loads(l) for l in (args.bench_root / "all_test.jsonl").open()]
    print(f"Loaded {len(all_rows)} rows from all_test.jsonl")

    def prompt(r): return r.get("prompt", "") or ""

    pam, me, cmi, ma = [], [], [], []
    for r in all_rows:
        src = r.get("source", "")
        if "PAM" in src:
            pam.append({"audio_path": r["audio-path"], "prompt": prompt(r),
                        "musicality_mos": float(r["musicality"])})
        elif "MusicEval" in src:
            me.append({"audio_path": r["audio-path"], "prompt": prompt(r),
                       "musicality_mos": float(r["musicality"])})
        elif "cmi-arena" in src:
            pf = r.get("preference-musicality", "")
            if pf in ("model_a", "model_b"):
                cmi.append({"audio_a": r["audio-path"], "audio_b": r["audio2"],
                            "prompt": prompt(r),
                            "winner": "a" if pf == "model_a" else "b"})
        elif "Music Arena" in src:
            p = r.get("preference", "").upper()
            if p in ("A", "B"):
                ma.append({"audio_a": r["audio-path"], "audio_b": r["audio2"],
                           "prompt": prompt(r),
                           "winner": "a" if p == "A" else "b"})

    print(f"PAM: {len(pam)} clips")
    print(f"MusicEval: {len(me)} clips")
    print(f"CMI-Pref: {len(cmi)} pairs ({Counter(p['winner'] for p in cmi)})")
    print(f"Music Arena: {len(ma)} pairs ({Counter(p['winner'] for p in ma)})")

    out = {"pam": pam, "musiceval": me, "cmi_pref": cmi, "music_arena": ma}
    args.out.write_text(json.dumps(out))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
