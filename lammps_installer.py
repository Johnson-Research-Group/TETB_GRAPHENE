import subprocess 
import os
import shutil

def lammps_installer(lammps_dir):
    print("installing shared library mode LAMMPS")
    cwd = os.getcwd()
    if lammps_dir:
        subprocess.call("cd "+lammps_dir,shell=True)
    subprocess.call("git clone https://github.com/lammps/lammps.git",shell=True)
    subprocess.call("cd lammps; git checkout b8acd2e31de3d19ac0a4629ef168e183335bcc74",shell=True)

    subprocess.call("cd src",shell=True)
    subprocess.call("cp "+os.path.join(cwd,"TETB_GRAPHENE/parameters/pair_reg_dep_poly.*")+" .",shell=True)
    subprocess.call("make serial mode=shared")
    subprocess.call("make install-python",shell=True)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--lammps_dir',type=str,default='.')
    args = parser.parse_args()
    lammps_installer(args.lammps_dir)
