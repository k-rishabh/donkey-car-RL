#!/bin/bash
#SBATCH --partition=gpu          # Use high-performance GPU partition if available
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1      # Use 4 GPUs            
#SBATCH --job-name=gpu_run
#SBATCH --mem=32GB                   # Allocate sufficient memory
#SBATCH --cpus-per-task=8            # Increase CPU resources
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Activate the virtual environment
source /home/phalle.y/fai/vs/donkey-car-RL/donkeyvenv/bin/activate

# Navigate to the directory containing your script
cd /home/phalle.y/fai/vs/donkey-car-RL/RL_algorithms/PPO

# Load optimized libraries (optional)
module load intel-mkl
module load cuda/11.8

# Execute your Python script
python ppo_train_final.py --sim /home/phalle.y/fai/projects/DonkeySimLinux/donkey_sim.x86_64 --env_name donkey-generated-roads-v0
