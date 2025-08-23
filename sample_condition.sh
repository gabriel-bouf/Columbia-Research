#!/bin/bash

#SBATCH --account=stats         
#SBATCH --job-name=DPS     
#SBATCH -c 1                      # The number of cpu cores to use
#SBATCH -t 0-2:30                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core
#SBATCH --account=stats
#SBATCH --nodes=1                  #nb of nodes
#SBATCH --gres=gpu:1
 
module load anaconda
source /burg-archive/opt/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate DPS2
echo "Environnement actif : $CONDA_DEFAULT_ENV"

#the main
#TASKCONFIG="configs/super_resolution_config.yaml"
TASKCONFIG="configs/dct_speckle_config.yaml"

MODELCONFIG="configs/model_config.yaml"

python3 sample_condition.py --model_config=$MODELCONFIG --diffusion_config=configs/diffusion_config.yaml --task_config=$TASKCONFIG;
echo ".sh executeee"

#run DPS with python3 sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml --task_config=configs/dct_speckle_config.yaml
