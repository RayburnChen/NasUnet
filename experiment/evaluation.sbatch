#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Ruibin Chen,  Apr 2021

# NOTE: To activate a SLURM option
# remove the whitespace between the '#' and 'SBATCH'

# Set your account id
#SBATCH --account=az20

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=5-00:00:00

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=RuibinJob

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Request for GPU, 
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

# SBATCH --gres=gpu:P100:1
# SBATCH --partition=m3h

# SBATCH --gres=gpu:1

# Memory usage (MB)
#SBATCH --mem-per-cpu=16000

# To receive an email when job completes or fails
#SBATCH --mail-user=rche0046@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=UNET-DARTS-EVA-JOB-%j.out

# Set the file for error log (stderr)
#SBATCH --error=UNET-DARTS-EVA-JOB-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:



# module load anaconda
module load anaconda/2019.03-Python3.7-gcc5
sleep 5
python -V

export PROJECT=az20
export CONDA_ENVS=/projects/$PROJECT/$USER/conda_envs
source activate $CONDA_ENVS/ruibinEnv
python -V

#conda install --file requirements.txt

sleep 5
python ./train.py --config ../configs/nas_unet/nas_unet_promise12.yml --model nasunet --ft
#python ./train.py --config ../configs/nas_unet/nas_unet_promise12.yml --model nasunet --ft --loss dice_sq_ce --genotype "Genotype(down=[('down_dep_conv', 1), ('down_dil_conv', 0), ('down_conv', 1), ('max_pool', 0), ('down_conv', 1), ('down_dep_conv', 0)], down_concat=range(2, 5), up=[('dil_conv', 0), ('up_dil_conv', 1), ('up_cweight', 1), ('dil_conv', 2), ('up_dil_conv', 1), ('shuffle_conv', 0)], up_concat=range(2, 5))"

nvidia-smi
# deviceQuery
