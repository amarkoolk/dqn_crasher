#!/bin/bash
#
#SBATCH --job-name=uniform_model_pool             # job name
#SBATCH --output=logs/%x_%j.out          # %x=job-name, %j=jobID
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu                   # your GPU partition
#SBATCH --gres=gpu:1                      # one GPU for the whole job
#SBATCH --constraint="h100_94gb|a40|a100_40gb|a100_80gb|a4000|a4500|a16|rtx_6000|rtx_2080ti"
#SBATCH --cpus-per-task=16               # total CPUs—change to suit (e.g. 4 runs × 4 cpus each)
#SBATCH --mem=64G                        # enough RAM for all runs together
#SBATCH --time=4-00:00:00

module purge
module load anaconda3
source activate safetyh

python src/dqn_crasher/safety_hardening_pool.py uniform_model_pool 5
