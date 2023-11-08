#!/bin/bash

echo "Starting scheduler..."

scheduler_file=$SCRATCH/scheduler_file.json
rm -f $scheduler_file

module load python/3.9-anaconda-2021.11
source activate $PSCRATCH/mypythonev

#start scheduler
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
dask-scheduler \
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
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
srun dask-worker \
--scheduler-file $scheduler_file \
    --interface hsn0 \
    --nworkers 1 
srun python mpi_ase_calc.py
echo "Killing scheduler"
kill -9 $dask_pid


