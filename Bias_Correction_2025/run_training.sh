#!/bin/bash
#SBATCH --job-name=bias_correction
#SBATCH --output=bias_correction_%j.out
#SBATCH --error=bias_correction_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Alternative PBS script (uncomment if using PBS instead of SLURM):
# #PBS -N bias_correction
# #PBS -o bias_correction_$PBS_JOBID.out
# #PBS -e bias_correction_$PBS_JOBID.err
# #PBS -l walltime=24:00:00
# #PBS -l nodes=1:ppn=4:gpus=1
# #PBS -l mem=32gb

echo "🚀 Starting Bias Correction Training on Supercomputer"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "=================================================="

# Load required modules (adjust for your system)
module load python/3.8
module load cuda/11.0
module load pytorch/1.8.0

# Or if using conda:
# source activate your_conda_env

# Check GPU
nvidia-smi

# Install required packages if needed
pip install xarray netcdf4 tqdm matplotlib

# Run the training script
python train_on_supercomputer.py

echo "=================================================="
echo "Training completed at: $(date)"
echo "Check the output files for results"
echo "=================================================="
