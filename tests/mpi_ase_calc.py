from mpi4py import MPI
import numpy as np
from ase import Atoms
import ase.io
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
import joblib
from joblib import Parallel, delayed
#from ase.units import units
import dask
import dask.distributed as dd
import os
import cupy as cp
from ase.parallel import world
import sys
from lammps import PyLammps


# Define a custom ASE calculator that performs parallel force calculations.
class ParallelForceCalculator(Calculator):
    implemented_properties = ['forces','energy','potential_energy']

    def __init__(self):
        super(ParallelForceCalculator, self).__init__()
        self.comm = MPI.COMM_WORLD 
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def init_pylammps(self,atoms):
        """ create pylammps object and calculate corrective potential energy
        """
        ntypes = len(set(atoms.get_chemical_symbols()))
        data_file = "tegt.data"
        ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
        L = PyLammps()
        L.command("units		metal")
        L.command("atom_style	full")
        L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
        L.command("box tilt large")

        L.command("read_data "+data_file)

        L.command("group top type 1")
        L.command("mass 1 12.0100")

        if ntypes ==2:
           L.command("group bottom type 2")
           L.command("mass 2 12.0100")

        L.command("velocity	all create 0.0 87287 loop geom")
        # Interaction potential for carbon atoms
        ######################## Potential defition ########################
        L.command("pair_style       rebo")
        L.command("pair_coeff      * * CH.rebo C")

        ####################################################################

        L.command("timestep 0.00025")
        L.command("thermo 1")
        L.command("fix 1 all nve")
        return L

    def run_lammps(self,atoms):
        """ evaluate corrective potential energy, forces in lammps
        """
        if not atoms.has("mol-id"):
            mol_id = np.ones(len(atoms),dtype=np.int8)
            sym = atoms.get_chemical_symbols()
            top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
            mol_id[top_layer_ind] += 1
            atoms.set_array("mol-id",mol_id)
        #update atom positions in lammps object, need to make sure pylammps object is only initialized on rank 0 so I don't have to keep writing data files
        #if not self.pylammps_started:
        self.L = self.init_pylammps(atoms)
        #pos = atoms.positions
        #for i in range(atoms.get_global_number_of_atoms()):
        #    self.L.atoms[i].position = pos[i,:]

        forces = np.zeros((atoms.get_global_number_of_atoms(),3))

        self.L.run(0)
        pe = self.L.eval("pe")
        ke = self.L.eval("ke")
        for i in range(atoms.get_global_number_of_atoms()):
            forces[i,:] = self.L.atoms[i].force
        del self.L
        return forces,pe

    def calculate(self, atoms, properties, system_changes):
        # You need to implement your force calculation here.
        # Perform calculations specific to your model.

        # For this example, calculate random forces.
        print("before tb calc ",world.rank)
        forces_k,energy_k = self.run_energy(len(atoms))
        #MPI.COMM_WORLD.barrier()
        #energy_k = world.comm.gather(energy_k,root=0)
        #forces_k = world.comm.gather(forces_k,root=0)
        # Communicate forces across processes
        if self.size > 1:
            print("gathering ",self.rank)
            forces_k = self.comm.gather(forces_k, root=0)
            energy_k = self.comm.gather(energy_k,root=0)
        else:
            print("gathering ",self.rank)
            forces_k = [forces_k]
            energy_k = energy_k
        
        #print("after tb calc ",world.rank)
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            # If you're on the root process, set the results in the atoms object. 
            potential_energy = np.sum(energy_k) #+lammps_potential_energy
            forces = np.sum(forces_k, axis=0) #+lammps_forces
            #self.results = {'energy': potential_energy, 'forces': forces, 'potential_energy':potential_energy}
            #print("rank 0 ",forces)

        else:
            potential_energy=None
            forces = None

        MPI.COMM_WORLD.barrier()
        #everything works up to here
        print("broadcasting rank ",self.rank)
        forces = MPI.COMM_WORLD.bcast(forces,root=0)
        potential_energy = MPI.COMM_WORLD.bcast(potential_energy,root=0)
        Lammps_forces, Lammps_potential_energy = self.run_lammps(atoms)
        print("forces = ",forces)
        print("energy = ",potential_energy)
        #print(type(forces))
        #atoms.arrays['forces'] = forces
        #atoms.arrays['energy'] = potential_energy
        #atoms.arrays['potential_energy'] = potential_energy
        self.results['forces'] = forces + Lammps_forces
        self.results['potential_energy'] = potential_energy + Lammps_potential_energy
        self.results['energy'] = potential_energy + Lammps_potential_energy
        print("tight binding finished")
        MPI.COMM_WORLD.barrier()
    
    def calculate_properties(self, atoms, results):
        potential_energy = np.sum([result['energy'] for result in results])
        forces = np.sum([result['forces'] for result in results], axis=0)
        return {'energy': potential_energy, 'forces': forces,'potential_energy':potential_energy}

    def tb_fxn(self,i):
        nkp = 4
        k_points = np.ones((nkp,3))
        #print("calculating index = ",i)
        n = 20
        ham = np.eye(n)*np.linalg.norm(k_points[i])
        #hamiltonian_gpu = cp.asarray(ham)

        # Step 3: Diagonalization on GPUs
        #eigenvalues, eigenvectors = cp.linalg.eigh(hamiltonian_gpu)

        # Step 4: Transfer Eigenvalues Back to CPUs
        #eigenvalues_cpu = cp.asnumpy(eigenvalues)
        #eigenvectors_cpu = cp.asnumpy(eigenvectors)
        eigenvalues_cpu,eigenvectors_cpu = np.linalg.eigh(ham)
        #print("diagonalized index = ",i)
        return eigenvalues_cpu,eigenvectors_cpu
    def run_energy(self,n):
        number_of_cpu = joblib.cpu_count()
        #kind = np.array(range(self.nkp))
        nkp = 4
        data = np.arange(nkp)
        local_data = data[world.rank::world.comm.size]
        output = Parallel(n_jobs=len(local_data))(delayed(self.tb_fxn)(i) for i in local_data)
        #band_data = comm.gather(output, root=0)
        #eigvals = np.zeros((20,nkp))
        #for i in range(nkp):
        #    eigvals[:,i] = output[i]
        eigvals = np.zeros((20,nkp))
        dim1 = len(output)
        #for i,bd_dim1 in enumerate(band_data):
        #    for j,bd_dim2 in enumerate(bd_dim1):
        #        eigvals[:,int(i*j+j)] = bd_dim2[0]
        e = 0
        f = np.zeros((n,3))
        
        for i in range(dim1):
            e += sum(output[i][0])
            f += np.sum(np.array(output[i][1]))
        return f,e

if __name__ == '__main__':
    # Create a custom parallel calculator
    calc = ParallelForceCalculator()

    # Create a simple test system with ASE
    atoms = Atoms('C2', positions=[(0, 0, 0), (0, 0, 0.74)],
                  cell = 10*np.eye(3))
    atoms.set_calculator(calc)

    # Molecular dynamics setup
    dyn = FIRE(atoms,
                   trajectory="test.traj",
                   logfile="test.log")
    dyn.run(fmax=0.00005)
    #nsteps = 100

    #dyn.run(nsteps)
    # Access the calculated forces after the simulation
    forces = atoms.get_forces()
    print("Forces at the end of the simulation:")
    print(forces)

