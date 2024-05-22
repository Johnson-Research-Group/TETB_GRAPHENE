import numpy as np
import generate_ensemble_models
import subprocess 

def submit_batch_file_delta(executable,batch_options,
                                 conda_env='myenv'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+'='+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")

        f.write("\nsource activate "+conda_env+"\n")
        #f.write("export OMP_NUM_THREADS=1\n")
        #f.write("type python\ntype dask\nscheduler_file=$HOME/scheduler_file.json\nrm -f $scheduler_file\nDASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=1000s \\" + "\nDASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=1000s \\"+"\ndask scheduler \\"+ "\n    --interface hsn0 \\"+ "\n    --scheduler-file $scheduler_file &\ndask_pid=$!\nsleep 5\nuntil [ -f $scheduler_file ]\ndo\n     sleep 5\ndone\nDASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=1000s \\"+ "\nDASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=1000s \\" +"\nsrun dask cuda worker \\" +"\n--scheduler-file $scheduler_file \\"+ "\n    --interface hsn0 \\" +"\n &\nsleep 5\n")
   
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)

if __name__=="__main__":
    batch_options_delta = {
            "--nodes":"1",
            "--time":"48:00:00",
            "--account":"bcmp-delta-cpu",
            #"--partition":"gpuA100x4,gpuA40x4",
            #"--partition":"gpuA100x8",
            "--job-name":"prod",
            #"--gpus-per-task":"1",
            "--cpus-per-task":"60",
            "--mem":"208g",
            "modules":['anaconda3_gpu/23.9.0']}
    
    output_file = "Hessian_matrix.txt"
    n_params = 58
    indi,indj = np.meshgrid((np.arange(n_params),np.arange(n_params)))
    for i in range(n_params):
        executable = "python generate_ensemble_models.py -m Hessian -o "+output_file+" -i "+str(i)
        submit_batch_file_delta(executable,batch_options_delta)