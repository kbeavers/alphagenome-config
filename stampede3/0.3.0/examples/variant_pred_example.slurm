#!/bin/bash
#----------------------------------------------------------------------
# Example AlphaGenome batch job for TACC Stampede3 (H100 / x86_64)
#
# It loads the alphagenome module and runs a Python analysis script on an
# H100 GPU node via `run_alphagenome`. AlphaGenome has no CLI — you write a
# short Python script that builds a dna_model and calls predict_variant /
# predict_interval / score_variant / etc.
# See `module help alphagenome` for details.
#----------------------------------------------------------------------
#SBATCH -J variant_pred           # job name
#SBATCH -o variant_pred.%j.out    # stdout -> variant_pred.<jobid>.out
#SBATCH -e variant_pred.%j.err    # stderr -> variant_pred.<jobid>.err
#SBATCH -p h100                   # Stampede3 H100 GPU queue
#SBATCH -N 1                      # nodes (4x H100 per node)
#SBATCH -n 1                      # tasks
#SBATCH -t 01:00:00               # walltime HH:MM:SS
#SBATCH -A your-project           # allocation / project name

module use /scratch/tacc/apps/bio/alphagenome/modulefiles
module load alphagenome/0.3.0-ctr

# --- Your own model weights (download once; accept the model terms) ----------
#   https://deepmind.google.com/science/alphagenome/model-terms
# The directory must contain the fold subdirs: all_folds/ fold_0/ ... fold_3/
export AG_MODELS_DIR=$WORK/alphagenome/models

# --- Run your analysis -------------------------------------------------------
# Point run_alphagenome at YOUR script. variant_pred.py is a working example
# that reads $AG_MODELS_DIR / $AG_REFERENCE_DIR and writes pv.png. Replace it 
# with your own script when ready.
#
# The example uses a single GPU (jax.local_devices()[0]). An H100 node has 4
# GPUs; request a full node above and select devices in your script if you want
# to use more than one.
cp $AG_EXAMPLES_DIR/variant_pred.py .
run_alphagenome variant_pred.py

# Outputs are written to the directory you submitted from (e.g. pv.png).
