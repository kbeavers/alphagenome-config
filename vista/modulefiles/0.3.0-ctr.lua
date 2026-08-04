local help_message = [[
This is a module file for the AlphaGenome container (kbeavers/alphagenome:0.3.0),
which exposes the following command:

 - run_alphagenome <script.py> [args...]

This runs your Python script with the AlphaGenome API inside an Apptainer container
using GPU support (--nv). AlphaGenome has no CLI; you write a small Python script that
builds a dna_model and calls predict_variant / predict_interval / score_variant, etc.

>> YOU MUST PROVIDE YOUR OWN MODEL WEIGHTS <<
Model weights are gated by a use agreement, so they cannot be shared centrally. Each
user downloads their own 5 folds (all_folds, fold_0..3) from Kaggle or Hugging Face
after accepting the model terms:
    https://deepmind.google.com/science/alphagenome/model-terms
Then point AG_MODELS_DIR at the directory that contains those fold subdirectories:
    export AG_MODELS_DIR=$WORK/alphagenome/models

Variables the module sets for you:
    AG_REFERENCE_DIR   -> shared genome reference files (fasta, gtf, feather, calibration)
    AG_EXAMPLES_DIR    -> example scripts (variant_pred.py, sbatch templates)
    AG_IMAGE           -> path to the .sif image

Variable YOU must set:
    AG_MODELS_DIR      -> your own downloaded weights (see above)

Example:
    module use /scratch/tacc/apps/bio/alphagenome/modulefiles
    module load alphagenome/0.3.0-ctr
    export AG_MODELS_DIR=$WORK/alphagenome/models
    cp $AG_EXAMPLES_DIR/variant_pred.py .
    run_alphagenome variant_pred.py      # writes pv.png

Optional (GH200 unified memory: let the GPU spill to host RAM for large sequences):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TF_FORCE_UNIFIED_MEMORY=true
    export XLA_CLIENT_MEM_FRACTION=3.2

Source code: https://github.com/google-deepmind/alphagenome_research
Built for TACC GPUs with a one-line jax[cuda13] override (+ tensorflow-cpu on x86_64).]]

help(help_message, "\n")

whatis("Name: alphagenome")
whatis("Version: 0.3.0")
whatis("Category: Bioinformatics")
whatis("Keywords: Container, AlphaGenome, JAX, genomics, variant-effect")
whatis("Description: AlphaGenome run environment using a TACC container image (Vista/GH200).")
whatis("URL: https://github.com/google-deepmind/alphagenome_research")

-- Environment vars (EDIT base path / version if different)
-- NOTE: AG_MODELS_DIR is intentionally NOT set here — weights are per-user (use agreement).
setenv("AG_HOME",          "/scratch/tacc/apps/bio/alphagenome/0.3.0")
setenv("AG_IMAGE",         "/scratch/tacc/apps/bio/alphagenome/0.3.0/image/alphagenome_0.3.0.sif")
setenv("AG_REFERENCE_DIR", "/scratch/tacc/apps/bio/alphagenome/0.3.0/reference")
setenv("AG_EXAMPLES_DIR",  "/scratch/tacc/apps/bio/alphagenome/0.3.0/examples")

-- Dependencies
-- NOTE: no system CUDA module is loaded. CUDA is bundled inside the image via
-- jax[cuda13]; apptainer's --nv injects the host driver at run time.
always_load("tacc-apptainer")

-- run_alphagenome: execute a user Python script inside the container on the GPU.
-- The wrapper first checks that the user set AG_MODELS_DIR (their own weights). The
-- shared reference dir is bound explicitly; the user's models dir is bound too. On
-- TACC, /scratch, /work and /home are auto-mounted, so weights under $SCRATCH/$WORK/$HOME
-- are visible. Relative outputs (e.g., pv.png) land in the launch directory.
set_shell_function("run_alphagenome",
"if [ -z \"$AG_MODELS_DIR\" ]; then " ..
"  echo >&2 'run_alphagenome: AG_MODELS_DIR is not set.'; " ..
"  echo >&2 'Download your own AlphaGenome weights (accept the model terms) and run:'; " ..
"  echo >&2 '    export AG_MODELS_DIR=/path/to/your/models'; " ..
"  echo >&2 'See: module help alphagenome'; " ..
"  return 1; " ..
"fi; " ..
-- --cleanenv keeps the host environment (e.g. an active conda base, or a loaded
-- cuda module) from leaking its LD_LIBRARY_PATH into the container and shadowing
-- the image's own CUDA 13 libraries. --nv still injects the GPU driver. We
-- re-pass only what's needed; CUDA_VISIBLE_DEVICES is forwarded when SLURM sets it.
"apptainer exec --nv --cleanenv " ..
"  ${CUDA_VISIBLE_DEVICES:+--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES } " ..
"  ${XLA_PYTHON_CLIENT_PREALLOCATE:+--env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE } " ..
"  ${TF_FORCE_UNIFIED_MEMORY:+--env TF_FORCE_UNIFIED_MEMORY=$TF_FORCE_UNIFIED_MEMORY } " ..
"  ${XLA_CLIENT_MEM_FRACTION:+--env XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION } " ..
"  --env AG_MODELS_DIR=$AG_MODELS_DIR " ..
"  --env AG_REFERENCE_DIR=$AG_REFERENCE_DIR " ..
"  --bind $AG_REFERENCE_DIR " ..
"  --bind $AG_MODELS_DIR " ..
"  $AG_IMAGE /opt/venv/bin/python \"$@\"",
-- C-shell version
"if ( ! $?AG_MODELS_DIR ) then echo 'run_alphagenome: set AG_MODELS_DIR to your downloaded weights (module help alphagenome)'; exit 1; endif; " ..
"apptainer exec --nv --cleanenv " ..
"  --env AG_MODELS_DIR=$AG_MODELS_DIR " ..
"  --env AG_REFERENCE_DIR=$AG_REFERENCE_DIR " ..
"  --bind $AG_REFERENCE_DIR " ..
"  --bind $AG_MODELS_DIR " ..
"  $AG_IMAGE /opt/venv/bin/python $*")
