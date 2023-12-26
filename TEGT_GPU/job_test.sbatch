#!/bin/bash
#SBATCH --constraint gpu
#SBATCH --output tb_MK_bands_2_88.log
#SBATCH --job-name tb_MK_bands
#SBATCH --qos overrun
#SBATCH --account m4205
#SBATCH --time 12:00:00
#SBATCH --nodes 1
module load python/3.9-anaconda-2021.11
module load julia
module load cudatoolkit/11.7
module load cpe-cuda/23.03
source activate $PSCRATCH/mypythonev
srun -N1 --ntasks=1 --gpus=1 python test_relax.py
