import joblib
from joblib import Parallel, delayed
import cupy as cp
from mpi4py import MPI
import numpy as np

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

def get_energy():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("rank = ",rank," size = ",comm.Get_size())
    number_of_cpu = joblib.cpu_count()
    #kind = np.array(range(self.nkp))
    nkp = 25
    data = np.arange(25)
    local_data = data[rank::comm.size]
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
    f = 0
    for i in range(dim1):
        e += sum(output[i][0])
        f += sum(output[i][1])
    return e,f

"""def tb_fxn(i):
    nkp = 25
    k_points = np.ones((nkp,3))
    n = 20
    ham = np.eye(n)*np.linalg.norm(k_points[i])

    # Step 3: Diagonalization on GPUs
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    return eigenvalues,eigenvectors

def get_energy():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    number_of_cpu = joblib.cpu_count()
    nkp = 25
    data = np.arange(25)
    local_data = data[rank::comm.size]
    print("kpoints per process = ",len(local_data)," on rank ",rank)
    output = Parallel(n_jobs=len(local_data))(delayed(tb_fxn)(i) for i in local_data)
    band_data = comm.gather(output, root=0)
    if rank == 0:
        energy = 0
        dim1 = len(band_data)
        for i,bd_dim1 in enumerate(band_data):
            for j,bd_dim2 in enumerate(bd_dim1):
                #print((i*j+j))
                #print(bd_dim2[0])
                energy += np.sum(bd_dim2[0])
        return energy"""

if __name__=="__main__":
    tsteps = 1
    for i in range(tsteps):
        energy_k,f_k = get_energy()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        energy = comm.gather(energy_k,root=0)
        if rank ==0:
            energy = sum(energy)
            print("index = ",i,energy)
        comm.bcast(energy,root=0)
        comm.barrier()
