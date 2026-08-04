# AlphaGenome container for TACC (Vista GH200 / Stampede3 H100)
#
# Builds from UPSTREAM google-deepmind/alphagenome_research (no fork). Two
# TACC/Hopper-specific changes: (1) a one-line JAX override — upstream pins a bare
# `jax` (CPU jaxlib), we install `jax[cuda13]<0.11.0` for the CUDA 13 GPU wheels
# (mirrors eriksf's `update-jax` branch); and (2) on x86_64 only, swap `tensorflow`
# for `tensorflow-cpu` so TF's GPU build doesn't poison JAX's CUDA init (see below).
#
# CUDA is provided by the pip wheels (jax[cuda13]) — NOT by the base image and
# NOT by a system `cuda` module. apptainer's `--nv` injects the host driver at
# run time, so the host just needs a CUDA-13-capable NVIDIA driver (verified on
# Vista GH200 and Stampede3 H100).
#
# Build MULTI-ARCH so one manifest serves both systems:
#   Vista     -> linux/arm64  (Grace-Hopper, aarch64)
#   Stampede3 -> linux/amd64  (H100 nodes, x86_64)
#
#   docker buildx build --platform linux/arm64,linux/amd64 \
#       --build-arg AG_REF=v0.3.0 \
#       -t kbeavers/alphagenome:0.3.0 --push .

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Build/runtime deps for the native extensions (pyBigWig, pyfaidx, pyarrow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python/env manager); mirrors the colleague's validated workflow.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install uv-managed Pythons under /opt (world-readable), NOT /root/.local (mode 700).
# Otherwise the venv's python symlinks into /root and apptainer, running as a
# non-root user, gets "stat: permission denied" traversing /root.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

# Don't keep a separate uv cache copy. The CUDA 13 wheels are several GB; caching
# them doubles peak build disk (the multi-arch build ran out of space extracting
# nvidia-nccl to /root/.cache/uv) and bloats the final image. Extract straight in.
ENV UV_NO_CACHE=1

# Precompile .pyc at build time. The container is read-only at run time, so without
# this Python recompiles the whole dependency tree (jax, tensorflow, pandas, …) on
# EVERY invocation and can't cache it — a large, repeated startup cost. Baking the
# bytecode into the image makes imports read precompiled .pyc instead.
ENV UV_COMPILE_BYTECODE=1

# --- Upstream code, pinned to a tag/commit for reproducibility ---
ARG AG_REF=v0.3.0
RUN git clone https://github.com/google-deepmind/alphagenome_research.git /opt/alphagenome_research \
    && cd /opt/alphagenome_research \
    && git checkout ${AG_REF} \
    && git rev-parse HEAD > /opt/AG_COMMIT      # record exact commit built

# --- Python 3.13 venv (matches what was validated at TACC) ---
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --seed --python 3.13 ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Apptainer does not reliably propagate the Docker image's PATH, and the base
# image has no `python`. Symlink the venv interpreter into /usr/local/bin, which
# IS on apptainer's default PATH, so both `python` and /opt/venv/bin/python work.
RUN ln -sf ${VIRTUAL_ENV}/bin/python /usr/local/bin/python \
    && ln -sf ${VIRTUAL_ENV}/bin/python /usr/local/bin/python3

# --- Install the package (pulls upstream's bare/CPU jax + all other deps) ---
RUN uv pip install --python ${VIRTUAL_ENV} -e /opt/alphagenome_research

# --- THE 'update-jax' DELTA: override bare jax with the CUDA 13 GPU wheels ---
# aarch64 + x86_64 wheels both exist for jax-cuda13-plugin. Keep the <0.11.0 cap.
RUN uv pip install --python ${VIRTUAL_ENV} "jax[cuda13]<0.11.0"

# Resolve the TensorFlow/JAX CUDA clash. On x86_64 the `tensorflow` wheel is
# GPU-enabled; when imported (via dna_model) it registers CUDA stubs that interpose
# on JAX's cuBLAS version check, so JAX's CUDA 13 plugin rejects it and silently
# falls back to CPU. Removing TF's cu12 pip libs is NOT enough — the GPU wheel
# still poisons the process. AlphaGenome uses TF only for data loading, so swap in
# the CPU-only TensorFlow build, which contains no CUDA and cannot interfere.
# tensorflow-cpu ships x86_64 wheels only; aarch64's `tensorflow` is already
# CPU-only, so gate the swap on architecture.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip uninstall --python ${VIRTUAL_ENV} tensorflow && \
        uv pip install --python ${VIRTUAL_ENV} tensorflow-cpu ; \
    fi

# Make everything under /opt readable + traversable by any (non-root) user, so
# apptainer can reach the interpreter, venv, and code when run as a normal user.
RUN chmod -R a+rX /opt

WORKDIR /work
CMD ["/opt/venv/bin/python"]
