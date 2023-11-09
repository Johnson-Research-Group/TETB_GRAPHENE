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
import sys
def tb_fxn(i):
    nkp = 4
    k_points = np.ones((nkp,3))
    print("calculating index = ",i)
    n = 20
    ham = np.eye(n)*np.linalg.norm(k_points[i])
    #hamiltonian_gpu = cp.asarray(ham)

    # Step 3: Diagonalization on GPUs
    #eigenvalues, eigenvectors = cp.linalg.eigh(hamiltonian_gpu)

    # Step 4: Transfer Eigenvalues Back to CPUs
    #eigenvalues_cpu = cp.asnumpy(eigenvalues)
    #eigenvectors_cpu = cp.asnumpy(eigenvectors)
    eigenvalues_cpu,eigenvectors_cpu = np.linalg.eigh(ham)
    print("diagonalized index = ",i)
    sys.stdout.flush() 
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
        print("before tb calc ",world.rank)
        forces_k,energy_k = self.run_tb(len(atoms))
        forces = np.mean(forces_k,axis=1)
        potential_energy = np.mean(energy_k)
        atoms.arrays['forces'] = forces
        atoms.arrays['energy'] = potential_energy
        atoms.arrays['potential_energy'] = potential_energy
        self.results['forces'] = forces
        self.results['potential_energy'] = potential_energy
        self.results['energy'] = potential_energy
        #world.comm.barrier()
        exit()
    
    def calculate_properties(self, atoms, results):
        potential_energy = np.sum([result['energy'] for result in results])
        forces = np.sum([result['forces'] for result in results], axis=0)
        return {'energy': potential_energy, 'forces': forces,'potential_energy':potential_energy}

    def run_tb(self,n):
        print("running tight binding ... \n\n\n\n")
        number_of_cpu = joblib.cpu_count()
        #kind = np.array(range(self.nkp))
        nkp = 4
        data = np.arange(nkp)
        local_data = data #[world.rank::world.comm.size]
        output = Parallel(n_jobs=nkp)(delayed(tb_fxn)(i) for i in local_data)
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

