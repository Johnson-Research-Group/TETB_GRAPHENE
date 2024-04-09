#import TEGT
import subprocess
from datetime import datetime

def submit_batch_file_perlmutter(executable,batch_options,
                                 conda_env='$PSCRATCH/mypythonev'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")

        f.write("\nsource activate "+conda_env+"\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)

def submit_batch_file_uiuc_cc(executable,batch_options,
                                 conda_env='my.anaconda'):

    sbatch_file="job"+str(hash(datetime.now()) )+".sbatch"
    batch_copy = batch_options.copy()

    prefix="#SBATCH "
    with open(sbatch_file,"w+") as f:
        f.write("#!/bin/bash\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+' '+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")

        f.write("\nsource activate "+conda_env+"\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)
        

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
        f.write("export OMP_NUM_THREADS=1\n")
        f.write(executable)
    subprocess.call("sbatch "+sbatch_file,shell=True)


if __name__=="__main__":
    batch_options_perlmutter = {
            "--nodes":"1",
            "--time":"12:00:00",
            "--account":"m4205",
            "--qos":"regular",
            "--job-name":"fit_potential",
            "--constraint":"cpu",
            "modules":['python/3.9-anaconda-2021.11','module load cudatoolkit/11.7','module load cpe-cuda/23.03']}

    batch_options_uiuc_cc= {
                 '--partition':'qmchamm',
                 '--nodes':1,
                 '--ntasks':1,
                 '--cpus-per-task':40,
                 '--time':'72:00:00',
                 '--output':'logout.fit_interlayer',
                 '--job-name':'fit_interlayer',
                 '--mail-user':'dpalmer3@illinois.edu',
                 '--mail-type':'ALL',
                 'modules':['anaconda']
        }
    batch_options_delta = {
            "--nodes":"1",
            "--time":"48:00:00",
            "--account":"bcmp-delta-gpu",
            "--partition":"gpuA100x4,gpuA40x4",
            #"--partition":"gpuA100x8",
            "--job-name":"prod",
            "--gpus-per-task":"1",
            "--cpus-per-task":"1",
            "--ntasks-per-node":"1",
            "--mem":"208g",
            "modules":['anaconda3_gpu/23.9.0']}
    batch_options = batch_options_perlmutter
    tb_models = ["popov"]
    nkp = [0] #,1,225]
    for t in tb_models:
        for k in nkp:
            executable = "python fit_potentials.py -m "+t+" -t interlayer -k "+str(k)+" -g True"
            batch_options["--job-name"]="nkp"+str(k)+"_interlayer"
            batch_options["--output"]= "nkp"+str(k)+"_interlayer.log"
            #submit_batch_file_delta(executable,batch_options_delta)
            #submit_batch_file_perlmutter(executable,batch_options_perlmutter)
            #submit_batch_file_uiuc_cc(executable,batch_options)

            executable = "python fit_potentials.py -m "+t+" -t intralayer -k "+str(k)+" -g True" # -oz tb_weight"
            batch_options["--job-name"]="nkp"+str(k)+"_intralayer"
            batch_options["--output"]= "nkp"+str(k)+"_intralayer.log"
            print(executable)
            submit_batch_file_delta(executable,batch_options_delta)
            #submit_batch_file_perlmutter(executable,batch_options)
            #submit_batch_file_uiuc_cc(executable,batch_options)
