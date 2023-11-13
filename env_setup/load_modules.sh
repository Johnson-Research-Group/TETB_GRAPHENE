#!/bin/bash
module load cudatoolkit/11.7
module load cpe-cuda/23.03

module load python/3.9-anaconda-2021.11
module load julia
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
source activate $PSCRATCH/mypythonev
