#!/bin/bash
#----------------------------------------------------------------------
# Example AlphaGenome batch job for TACC Vista (GH200 / aarch64)
#
# Copy this file, edit the lines marked [EDIT], and submit with:
#     sbatch alphagenome_example.slurm
#
# It loads the alphagenome module and runs a Python analysis script on a
# Grace-Hopper GPU node via `run_alphagenome`. AlphaGenome has no CLI —
# you write a short Python script that builds a dna_model and calls
# predict_variant / predict_interval / score_variant / etc.
# See `module help alphagenome` for details.
#----------------------------------------------------------------------
#SBATCH -J alphagenome            # job name
#SBATCH -o alphagenome.%j.out     # stdout  -> alphagenome.<jobid>.out
#SBATCH -e alphagenome.%j.err     # stderr  -> alphagenome.<jobid>.err
#SBATCH -p gh                     # Vista GPU queue (Grace-Hopper)
#SBATCH -N 1                      # nodes (1 GH200 GPU per node)
#SBATCH -n 1                      # tasks
#SBATCH -t 01:00:00               # walltime HH:MM:SS            [EDIT as needed]
#SBATCH -A your-project           # allocation / project name    [EDIT]
## #SBATCH --mail-user=you@utexas.edu
## #SBATCH --mail-type=all

module use /scratch/tacc/apps/bio/alphagenome/modulefiles
module load alphagenome/0.3.0-ctr

# --- Your own model weights (download once; accept the model terms) ----------
#   https://deepmind.google.com/science/alphagenome/model-terms
# The directory must contain the fold subdirs: all_folds/ fold_0/ ... fold_3/
export AG_MODELS_DIR=$WORK/alphagenome/models 

# --- Optional (GH200 unified memory): let the GPU spill to host RAM for very
# --- long sequences. Uncomment if you hit out-of-memory errors.
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export TF_FORCE_UNIFIED_MEMORY=true
# export XLA_CLIENT_MEM_FRACTION=3.2

# --- Run your analysis -------------------------------------------------------
# Point run_alphagenome at YOUR script. variant_pred.py is a working example
# that reads $AG_MODELS_DIR / $AG_REFERENCE_DIR and writes pv.png. Replace it 
# with your own script when ready.
cp $AG_EXAMPLES_DIR/variant_pred.py .
run_alphagenome variant_pred.py

# Outputs are written to the directory you submitted from (e.g. pv.png).
