# Total Energy Tight Binding for Graphene. 
This module is written entirely in python and can calculate Forces, Energies, and Band structures for multi-layer graphene systems outlined in this paper (). 

This module depends on pylammps. run the following command from the command line to install pylammps with the correct version of lammps and the necessary potentials

```python lammps_installer.py -d [optional: directory to lammps] ```

# Directory structure:
 * TETB_GRAPHENE/ source code
 * data/ ase databases of ab initio data used for fitting residual potentials
 * fit_potentials/ code used to fit residual potentials
 * tests/example_usage.py contains examples on how to run:
    - relaxations
    - total energy calculations
    - band structure calculations

# GPU capabilities
The TETB_GRAPHENE_GPU branch contains the same code as the default branch, however, it is written using cupy in order to speed up calculations for large systems. 

```git checkout TETB_GRAPHENE_GPU```


