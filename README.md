# AlphaGenome @ TACC

Configuration for running [AlphaGenome](https://github.com/google-deepmind/alphagenome_research)
as an environment module on TACC's **Vista** (GH200, aarch64) and **Stampede3** (H100, x86_64).
A single multi-arch container holds Python + JAX(GPU) + the AlphaGenome code; per-version model
weights and reference files live outside the container; a modulefile exposes a `run_alphagenome`
command. Validated on both systems for the `0.3.0` build.

## Repo layout

This repo mirrors what gets installed on the HPC systems (config + examples only — the image,
reference data, and weights are not in git):

```
alphagenome-config/
├── Dockerfile                              # multi-arch image recipe (arm64 + amd64)
├── vista/
│   ├── modulefiles/0.3.0-ctr.lua           # Vista modulefile
│   └── 0.3.0/examples/                     # variant_pred.py, variant_pred_example.slurm
└── stampede3/
    ├── modulefiles/0.3.0-ctr.lua           # Stampede3 modulefile
    └── 0.3.0/examples/                     # variant_pred.py, variant_pred_example.slurm
```

The two systems are kept as separate trees because their modulefiles differ (help text, and the
Stampede3 apps path may diverge). The `run_alphagenome` wrappers are otherwise identical.

## How it works

- **Built from upstream, no fork.** The image clones `google-deepmind/alphagenome_research` at a
  pinned tag and applies exactly two TACC-specific changes in the Dockerfile:
  1. `jax[cuda13]<0.11.0` — upstream pins a bare (CPU) `jax`; we install the CUDA 13 GPU wheels.
  2. On **x86_64 only**, swap `tensorflow` → `tensorflow-cpu` (see Gotchas).
- **JAX does the GPU work; TensorFlow only reads data (on CPU).** Model compute, `predict_variant`,
  etc. are all JAX on the GPU. TF is a data-loading dependency and never needs the GPU here.
- **CUDA comes from the pip wheels, not the OS.** No `nvidia/cuda` base image, no `cuda` module.
  Apptainer's `--nv` injects the host driver at run time. **Host requirement: an NVIDIA driver
  that supports CUDA ≥ 13** (check with `nvidia-smi` → "CUDA Version" field on a GPU node).
- **One multi-arch image** serves both systems: Vista pulls `linux/arm64`, Stampede3 pulls
  `linux/amd64`, from the same tag.
- **Weights are per-user; reference data is shared.** Model weights are gated by a use agreement,
  so the module does **not** set `AG_MODELS_DIR` — each user downloads their own and exports it.

## Build & push the image

From a machine with Docker + buildx (not a TACC login node):

```bash
docker login
docker buildx create --name agbuilder --use --bootstrap   # once

docker buildx build --platform linux/arm64,linux/amd64 \
  --build-arg AG_REF=v0.3.0 \
  -t kbeavers/alphagenome:0.3.0 --push .
```

If the multi-arch build runs out of disk (the CUDA 13 wheels are several GB × 2 arches), build
one arch at a time and stitch them:

```bash
docker buildx build --platform linux/arm64 --build-arg AG_REF=v0.3.0 -t kbeavers/alphagenome:0.3.0-arm64 --push .
docker buildx build --platform linux/amd64 --build-arg AG_REF=v0.3.0 -t kbeavers/alphagenome:0.3.0-amd64 --push .
docker buildx imagetools create -t kbeavers/alphagenome:0.3.0 \
  kbeavers/alphagenome:0.3.0-arm64 kbeavers/alphagenome:0.3.0-amd64
```

When finished, clean up with:

```bash
docker buildx prune -af
docker system prune -af
```

## Deploy / update to a new version

On-HPC layout (admin-managed; weights are per-user and NOT here):

```
/scratch/tacc/apps/bio/alphagenome/
├── modulefiles/<version>-ctr.lua
└── <version>/
    ├── image/alphagenome_<version>.sif
    ├── reference/            # ~6.3 GB shared genome files
    └── examples/
```

Steps for a new `<version>` (e.g. built from upstream tag `vX.Y.Z`):

1. **Check the upstream diff** — confirm the only change still needed is the `jax` line
   (and TF swap). Bump `AG_REF` in the Dockerfile to the new tag.
   ```bash
   git -C alphagenome_research diff <old_tag> <new_tag> -- pyproject.toml
   ```
2. **Build + push** the multi-arch image (above).
3. **Per system:** create `<version>/{image,reference,examples}`, then
   `module load tacc-apptainer && apptainer pull --force <version>/image/alphagenome_<version>.sif docker://kbeavers/alphagenome:<version>`.
   Symlink `reference/` to the previous version if unchanged:
   `ln -s ../<old>/reference <version>/reference`.
4. **Copy examples** into `<version>/examples/` and add `modulefiles/<version>-ctr.lua`
   (copy an existing one; update the version string and the four `setenv` paths).
5. **Validate** in an `idev` session on a GPU node (see "Validate the install" below).
   Success = `pv.png`, no cuBLAS error, assert passes.
6. **Commit** the new modulefile + examples here; note the exact upstream commit built.

## Use it

```bash
module use /scratch/tacc/apps/bio/alphagenome/modulefiles
module load alphagenome/0.3.0-ctr

# Your own weights (download the 5 folds, accept the model terms first):
#   https://deepmind.google.com/science/alphagenome/model-terms
export AG_MODELS_DIR=$WORK/alphagenome/models

cp $AG_EXAMPLES_DIR/variant_pred.py .
run_alphagenome variant_pred.py        # writes pv.png
```

`run_alphagenome <script.py>` runs the script with `python` inside the container with `--nv`
(GPU) and `--cleanenv` (no host-env leakage), binding the shared reference dir and your models
dir. For batch jobs, copy `variant_pred_example.slurm` (Vista queue `gh`, Stampede3 `h100`) and
edit it. The example script asserts it's on a GPU up front, so a silent CPU fallback fails loudly.

## Gotchas baked into the Dockerfile (why each line exists)

Each of these was a real failure during bring-up — don't remove them without understanding why:

| Setting / step | Problem it solves |
| --- | --- |
| `UV_PYTHON_INSTALL_DIR=/opt/uv/python` | Default puts the interpreter under `/root` (mode 700); apptainer runs as a non-root user → `stat: permission denied`. |
| `chmod -R a+rX /opt` (final layer) | Makes the venv, interpreter, and code traversable by any user at run time. |
| Symlink venv python into `/usr/local/bin` + call `/opt/venv/bin/python` in the wrapper | Apptainer doesn't reliably propagate the image `PATH`; bare `python` else "not found". |
| `UV_NO_CACHE=1` | CUDA 13 wheels are multi-GB; the cache doubled build disk (multi-arch OOM) and bloated the image. |
| `UV_COMPILE_BYTECODE=1` | Read-only container can't cache `.pyc`; without this Python recompiles jax/TF on every run (~2x startup). |
| `tensorflow-cpu` on x86_64 (`TARGETARCH`) | The GPU `tensorflow` wheel, imported before JAX, registers CUDA stubs that break JAX's cuBLAS check → silent CPU fallback. CPU-only TF has no CUDA and can't interfere. No-op on aarch64 (already CPU-only there). |
| `run_alphagenome` uses `--cleanenv` | Prevents a host env (e.g. active conda `base`) from leaking `LD_LIBRARY_PATH` and shadowing the image's CUDA libs. `--nv` still injects the driver; `CUDA_VISIBLE_DEVICES` is re-forwarded. |

## Requirements

- Host NVIDIA driver supporting **CUDA ≥ 13** (Vista GH200 and Stampede3 H100 both qualify).
- `tacc-apptainer` module (loaded automatically by the modulefile).
- A Docker Hub namespace + `buildx` to build/push images.
- Per user: a Kaggle/Hugging Face account with the AlphaGenome model terms accepted, to download
  the weights.
