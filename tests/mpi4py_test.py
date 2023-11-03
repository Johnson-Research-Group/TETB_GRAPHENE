import flatgraphene as fg
from mpi4py import MPI
import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
from mpi4py import MPI

def run_lammps(atoms):
    #update atom positions in lammps object
    L = init_pylammps(atoms)
    pos = atoms.positions
    for i in range(atoms.get_global_number_of_atoms()):
        L.atoms[i].position = pos[i,:]
    forces = np.zeros((atoms.get_global_number_of_atoms(),3))
    
    L.run(0)
    pe = L.eval("pe")
    ke = L.eval("ke")
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force
    return forces,pe,pe+ke

def init_pylammps(atoms_object):
        data_file = "tegt.data"
        #ase.io.write(data_file,atoms_object,format="lammps-data",atom_style = "full")
        L = PyLammps()
        L.command("units		metal")
        L.command("atom_style	full")
        L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
        L.command("box tilt large")
    
        L.command(" read_data "+data_file)
        L.command("group top type 1")
        L.command("group bottom type 2")
    
        L.command("mass 1 12.0100")
        L.command("mass 2 12.0100")
    
        L.command("velocity	all create 0.0 87287 loop geom")
        # Interaction potential for carbon atoms
        ######################## Potential defition ########################
    
        L.command("pair_style rebo")
        L.command("pair_coeff * * CH_pz.rebo C C")
        ####################################################################

        L.command("timestep 0.00025")
    
        L.command("thermo 1")
        L.command("fix 1 all nve")
        return L

def run_tight_binding(positions):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    natoms = np.shape(positions)[0]
    nkp=4
    k_points = np.ones((nkp,3))
    num_k_points = len(k_points)
    k_points_per_process = num_k_points // size
    my_k_points_start = rank * k_points_per_process
    my_k_points_end = (rank + 1) * k_points_per_process if rank < size - 1 else num_k_points

    # Construct local Hamiltonian matrices for assigned k-points
    local_eigenvalues = []
    for k_point in k_points[my_k_points_start:my_k_points_end]:
        local_hamiltonian = np.eye(natoms)*np.linalg.norm(k_point) #dummy hamiltonian
        
        # Diagonalize the local Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(local_hamiltonian)
        local_eigenvalues.append(eigenvalues)

    # Gather eigenvalues from all processes (optional)
    all_eigenvalues = comm.gather(local_eigenvalues, root=0)
    energy = sum(all_eigenvalues)
    return energy

#if __name__=="__main__":
t=21.78
a=2.46
sep=3.35
p_found, q_found, theta_comp = fg.twist.find_p_q(t)
atoms_object=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                    mass=[12.01,12.02],sep=sep,h_vac=20)
data_file = "tegt.data"
ase.io.write(data_file,atoms_object,format="lammps-data",atom_style = "full")
"""comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

positions = atoms_object.positions
positions = comm.bcast(positions, root=0)"""
rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print(rank, mpi_size)
"""if rank == 0:
    Lammps_forces,Lammps_potential_energy,Lammps_tot_energy= run_lammps(atoms_object)
else:
    print("running tight binding on processor "+str(rank))
    tb_Energy,tb_forces = run_tight_binding(positions)"""
