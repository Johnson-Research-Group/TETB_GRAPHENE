Total Energy Tight Binding for Graphene. 
This module is written entirely in python and can calculate Forces, Energies, and Band structures for multi-layer graphene systems outlined in this paper (). 

tests/test_relax.py contains examples on how to run relaxations, total energy calculations, or band structure calculations

The TETB_GRAPHENE_GPU contains the same code as the default branch, however, it is written using cupy in order to speed up calculations for large systems. 

Dependencies:
pylammps: https://docs.lammps.org/Howto_pylammps.html#step-1-building-lammps-as-a-shared-library\\
cp TETB_GRAPHENE/parameters/pair_reg_dep_poly.* ${LAMMPS_DIR}/lammps/src

