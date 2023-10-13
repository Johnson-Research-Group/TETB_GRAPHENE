import TEGT
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

if __name__=="__main__":
    batch_options = {
            "--nodes":"1",
            "--time":"12:00:00",
            "--account":"m4205",
            "--qos":"regular",
            "--job-name":"fit_potential",
            "--constraint":"cpu",
            "modules":['python/3.9-anaconda-2021.11']}

    tb_models = ["popov"]
    nkp = [225]
    for t in tb_models:
        for k in nkp:
            executable = "python fit_potentials.py -m "+t+"-t interlayer -k "+str(k)+" -s True"
            submit_batch_file_perlmutter(executable,batch_options)

            #executable = "python fit_potentials.py -m "+t+"-t intralayer -k "+str(k)+" -s True"
            #submit_batch_file_perlmutter(executable,batch_options)