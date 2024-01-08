
$LAMMPS_DIR = ${HOME}
cwd = pwd
$POTENTIAL_FILES = cwd/TEGT/parameters
source activate $CONDA_ENV
cd $LAMMPS_DIR; git clone https://github.com/lammps/lammps.git
cd lammps; git checkout b8acd2e31de3d19ac0a4629ef168e183335bcc74

cd src; cp $POTENTIAL_FILES/pair_reg_dep_poly.* .
make clean-all;
make yes-tally;
make yes-molecule;
make yes-manybody;
make yes-interlayer;
make -j20 mode=shared serial;
make install-python
#optionally enforce version
#make install-python PYTHON=python3.10
