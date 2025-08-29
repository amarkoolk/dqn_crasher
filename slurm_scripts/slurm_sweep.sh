#!/bin/bash
#
#SBATCH --job-name=dqn_sweep             # job name
#SBATCH --output=logs/%x_%j.out          # %x=job-name, %j=jobID
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu                   # your GPU partition
#SBATCH --gres=gpu:4                      # one GPU for the whole job
#SBATCH --cpus-per-task=16               # total CPUs—change to suit (e.g. 4 runs × 4 cpus each)
#SBATCH --mem=64G                        # enough RAM for all runs together
#SBATCH --time=4-00:00:00

module purge
module load anaconda3
source activate safetyh

# the sweep you want to run
SWEEP_ID="amar-research/sweep_10m_steps/oqlgsygl"

# how many parallel agents (and thus runs) to spin up on that one GPU
RUNS_PER_GPU=20
CUDA_VISIBLE_DEVICES=1

echo "Launching $RUNS_PER_GPU wandb agent instances on GPU $CUDA_VISIBLE_DEVICES"

for i in $(seq 1 $RUNS_PER_GPU); do
  echo " → starting agent #$i"
  # --count 1 makes each agent pull exactly one trial from the sweep
  wandb agent $SWEEP_ID --count 1 &
done

# wait for all of them to complete
wait
echo "All $RUNS_PER_GPU runs finished."
