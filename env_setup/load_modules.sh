#!/bin/bash
module load cudatoolkit/12.0
module load cpe-cuda/23.03

module load python/3.9-anaconda-2021.11
module load julia
source activate $PSCRATCH/mypythonev
