import subprocess 
import os
import shutil

def lammps_installer(lammps_dir):
    print("installing shared library mode LAMMPS")
    cwd = os.getcwd()
    subprocess.call("cd "+lammps_dir,shell=True)
    if not os.path.exists("lammps"):
        subprocess.call("git clone https://github.com/lammps/lammps.git",shell=True)
    
    os.chdir("lammps")
    subprocess.call("git checkout b8acd2e31de3d19ac0a4629ef168e183335bcc74",shell=True)

    os.chdir("src")
    subprocess.call("cp "+os.path.join(cwd,"TETB_GRAPHENE/parameters/pair_reg_dep_poly.*")+" .",shell=True)
    subprocess.call("make clean-all",shell=True)
    subprocess.call("make yes-tally",shell=True)
    subprocess.call("make yes-molecule",shell=True)
    subprocess.call("make yes-manybody",shell=True)
    subprocess.call("make yes-interlayer",shell=True)
    subprocess.call("make -j 16 serial mode=shared",shell=True)
    subprocess.call("make install-python",shell=True)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--lammps_dir',type=str,default='.')
    args = parser.parse_args()
    lammps_installer(args.lammps_dir)
