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
# Define a custom ASE calculator that performs parallel force calculations.
class ParallelForceCalculator(Calculator):
    implemented_properties = ['forces','energy','potential_energy']

    def __init__(self):
        super(ParallelForceCalculator, self).__init__()
        """self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()"""

    def calculate(self, atoms, properties, system_changes):
        # You need to implement your force calculation here.
        # Perform calculations specific to your model.

        # For this example, calculate random forces.
        forces,energy = self.run_tb(len(atoms))

        # Communicate forces across processes
        """if self.size > 1:
            forces = self.comm.gather(forces, root=0)
            energies = self.comm.gather(energy,root=0)
        else:
            forces = [forces]
            energies = energy

        if self.rank == 0:
            # If you're on the root process, set the results in the atoms object.
            atoms.arrays['forces'] = np.vstack(forces)
            atoms.arrays['energy'] = np.sum(energies)
            atoms.arrays['potential_energy'] = np.sum(energies)"""
        self.results['forces'] = forces
        self.results['potential_energy'] = energy
        self.results['energy'] = energy

    @dask.delayed
    def tb_fxn(self,i):
        nkp = 25
        n=20
        print("kpoint index = ",i)
        k_points = np.ones((nkp,3))
        ham = np.eye(n)*np.linalg.norm(k_points[i])

        # Step 3: Diagonalization on GPUs
        eigenvalues, eigenvectors = np.linalg.eigh(ham)

        return np.sum(eigenvalues),eigenvectors

    def reduce(self,results):
        energy = 0
        forces = np.zeros((2,3))
        for i,outk in results:
            energy +=outk[0]
            forces += np.array(outk[1])[:2,:3]
        return forces,energy

    def run_tb(self,n):
        nkp = 25
        kpoints = np.arange(25)
        cluster = dd.LocalCluster() #processes=True)
        #scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")
        #client = dd.Client(scheduler_file=scheduler_file)
        #client
        #client = dd.Client(n_workers=4)
        #client = dd.Client(cluster)
        #futures = [dask.delayed(self.tb_fxn)(kpoint) for kpoint in kpoints]
        #total = nkp
        #tasks = 8
        #count = total // tasks

        #futures = client.map(self.tb_fxn, kpoints, count=count)
        #forces,energy = client.submit(self.reduce, futures).result()
        energy = 0
        forces = np.zeros((n,3))
        results = []
        for i in kpoints:
            output = tb_fxn(i)
            results.append(output)
            #energy += results[0]
            #forces += np.array(results[1])[:n,:3]
        
        dask.compute(results)
        #client.close()
        #energy = 0
        #forces = np.zeros((n,3))
        for i in range(len(results)):
            energy+=results[i][0]
            forces+=np.array(results[i][1])[:n,:3]
        
        return forces,energy

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

