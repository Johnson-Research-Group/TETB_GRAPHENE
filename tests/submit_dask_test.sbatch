#!/bin/bash
#SBATCH --qos debug
#SBATCH --job-name test
#SBATCH --constraint cpu
#SBATCH --nodes=2
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=4
module load julia
module load cpu
#module load cudatoolkit/11.7
module load python/3.9-anaconda-2021.11
source activate $PSCRATCH/mypythonev
#export PATH=/pscratch/sd/d/dpalmer3/mypythonev/bin:$PATH

type python

type dask

echo "Starting scheduler..."

scheduler_file=$SCRATCH/scheduler_file.json
rm -f $scheduler_file

#start scheduler
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=1000s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=1000s \
dask scheduler \
    --interface hsn0 \
    --scheduler-file $scheduler_file &

dask_pid=$!

# Wait for the scheduler to start
sleep 5
until [ -f $scheduler_file ]
do
     sleep 5
done

echo "Starting workers"

#start scheduler
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=1000s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=1000s \
srun dask worker \
--scheduler-file $scheduler_file \
    --interface hsn0 \
    --nworkers 1 &

sleep 5
python test_relax.py
#python dask_calculate_pi.py
#echo "Killing scheduler"
#kill -9 $dask_pid

