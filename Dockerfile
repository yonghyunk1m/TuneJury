# Dockerfile — main `tunejury` environment (covers `docs/reproducing.md`
# §3 training, §4 internal eval, §10 anchor calibration, §11 probes).
#
# Scope: this image reproduces TuneJury training, scoring, paper figures,
# encoder-swap / leave-one-out / OOD eval (everything from
# `pip install -e .`). It does NOT include the Mode 1/2/3 backbone envs
# (musicgen, sao, audioldm2, acestep, tangoflux, meanaudio) — those pin
# incompatible torch / transformers versions and live in their own
# environments per `applications/mode*/README.md`.
#
# Build:
#   docker build -t tunejury:0.1 .
#
# Run (interactive, host GPU 3 for example, mount paper data + ckpts):
#   docker run --rm -it --gpus '"device=3"' \
#       -v $(pwd):/workspace -w /workspace tunejury:0.1 bash
#
# Inside the container:
#   python -c "from tunejury import Scorer; \
#       s = Scorer.from_pretrained('checkpoints/tunejury.pt'); \
#       print(s.score('release_scores/<some_wav>.wav', prompt=''))"

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git ffmpeg libsndfile1 build-essential ca-certificates \
        fluidsynth fluid-soundfont-gm \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Torch first so the resolver does not pull a CPU-only wheel after
# requirements pull in a torchaudio that constrains it. Versions pinned
# to match environment.yml (audit 2026-05-26): torch 2.7.1 on CUDA 12.6
# wheels, since 2.7.1 does not ship a CUDA 12.1 wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.7.1 torchaudio==2.7.1

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Repo itself is bind-mounted at runtime (see "Run" comment above);
# this final step makes `import tunejury` work without `pip install -e .`
# inside the container by adding the bind-mount path to PYTHONPATH.
ENV PYTHONPATH=/workspace

CMD ["bash"]
