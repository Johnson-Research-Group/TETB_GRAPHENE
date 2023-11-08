from mpi4py import MPI
import numpy as np
from ase import Atoms
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

def tb_fxn(i):
    nkp = 25
    k_points = np.ones((nkp,3))
    print("index = ",i)
    n = 20
    ham = np.eye(n)*np.linalg.norm(k_points[i])
    #hamiltonian_gpu = cp.asarray(ham)

    # Step 3: Diagonalization on GPUs
    #eigenvalues, eigenvectors = cp.linalg.eigh(hamiltonian_gpu)

    # Step 4: Transfer Eigenvalues Back to CPUs
    #eigenvalues_cpu = cp.asnumpy(eigenvalues)
    #eigenvectors_cpu = cp.asnumpy(eigenvectors)
    eigenvalues_cpu,eigenvectors_cpu = np.linalg.eigh(ham)
    return eigenvalues_cpu,eigenvectors_cpu



# Define a custom ASE calculator that performs parallel force calculations.
class ParallelForceCalculator(Calculator):
    implemented_properties = ['forces','energy','potential_energy']

    def __init__(self):
        super(ParallelForceCalculator, self).__init__()
        self.comm = MPI.COMM_WORLD 
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def calculate(self, atoms, properties, system_changes):
        # You need to implement your force calculation here.
        # Perform calculations specific to your model.

        # For this example, calculate random forces.
        forces_k,energy_k = self.run_tb(len(atoms))
        
        energy_k = world.comm.gather(energy_k,root=0)
        forces_k = world.comm.gather(forces_k,root=0)
        # Communicate forces across processes
        """if self.size > 1:
            forces = self.comm.gather(forces, root=0)
            energies = self.comm.gather(energy,root=0)
        else:
            forces = [forces]
            energies = energy"""

        if world.rank == 0:

            # If you're on the root process, set the results in the atoms object.
            potential_energy = np.sum(energy_k)
            forces = np.sum(forces_k, axis=0)
            self.results = {'energy': potential_energy, 'forces': forces, 'potential_energy':potential_energy}

        #else:
        #    total_energy=None
        #    forces = None

        #self.comm.barrier()
        #self.comm.bcast(forces,root=0)
        #self.comm.bcast(total_energy,root=0)
        #print(type(forces))
        #atoms.arrays['forces'] = forces
        #atoms.arrays['energy'] = total_energy
        #atoms.arrays['potential_energy'] = total_energy
        #self.results['forces'] = forces
        #self.results['potential_energy'] = total_energy
        #self.results['energy'] = total_energy
    
    def calculate_properties(self, atoms, results):
        potential_energy = np.sum([result['energy'] for result in results])
        forces = np.sum([result['forces'] for result in results], axis=0)
        return {'energy': potential_energy, 'forces': forces,'potential_energy':potential_energy}

    def run_tb(self,n):
        number_of_cpu = joblib.cpu_count()
        #kind = np.array(range(self.nkp))
        nkp = 25
        data = np.arange(25)
        local_data = data[self.rank::self.comm.size]
        output = Parallel(n_jobs=nkp)(delayed(tb_fxn)(i) for i in local_data)
        #band_data = comm.gather(output, root=0)
        #eigvals = np.zeros((20,nkp))
        #for i in range(nkp):
        #    eigvals[:,i] = output[i]
        eigvals = np.zeros((20,25))
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
    #comm = MPI.COMM_WORLD
    calc = ParallelForceCalculator()

    # Create a simple test system with ASE
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)])
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

