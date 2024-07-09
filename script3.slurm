#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=gpu
#SBATCH --time=00:03:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=tc064-s2603149
#SBATCH --mem=16G                # Adjust memory per node
#SBATCH --output=my_job_output.txt
#SBATCH --error=my_job_error.txt

module load python/3.9.13

# Define CONDA_ROOT
export CONDA_ROOT=/work/tc064/tc064/s2603149/condaenvs
export CONDARC=/work/tc064/tc064/s2603149/condaenvs/.condarc  # Set your specific CONDARC path

eval "$(conda shell.bash hook)"
conda activate /work/tc064/tc064/s2603149/condaenvs/envs/myvenv

export MASTER_PORT=12340
export WORLD_SIZE=16

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd tfm-try
python -m torch.distributed.launch --nproc_per_node=4 --master_port ${MASTER_PORT} train.py --exp_name CMT-final --data_path ../merged --epochs 50 --lr 2e-5 --batch_size 6 --num_workers 4 --pred_box
