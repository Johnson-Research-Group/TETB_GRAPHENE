import joblib
from joblib import Parallel, delayed
import cupy as cp
from mpi4py import MPI
import numpy as np
def tb_fxn(i):
    nkp = 25
    k_points = np.ones((nkp,3))
    n = 20
    ham = np.eye(n)*np.linalg.norm(k_points[i])
    hamiltonian_gpu = cp.asarray(ham)

    # Step 3: Diagonalization on GPUs
    eigenvalues, eigenvectors = cp.linalg.eigh(hamiltonian_gpu)

    # Step 4: Transfer Eigenvalues Back to CPUs
    eigenvalues_cpu = cp.asnumpy(eigenvalues)
    eigenvectors_cpu = cp.asnumpy(eigenvectors)
    return eigenvalues_cpu,eigenvectors_cpu

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    number_of_cpu = joblib.cpu_count()
    #kind = np.array(range(self.nkp))
    nkp = 25
    data = np.arange(25)
    local_data = data[rank::comm.size]
    print(rank,comm.size,local_data)
    output = Parallel(n_jobs=nkp)(delayed(tb_fxn)(i) for i in local_data)
    band_data = comm.gather(output, root=0)
    #eigvals = np.zeros((20,nkp))
    #for i in range(nkp):
    #    eigvals[:,i] = output[i]
    if rank == 0:
        eigvals = np.zeros((20,25))
        dim1 = len(band_data)
        for i,bd_dim1 in enumerate(band_data):
            for j,bd_dim2 in enumerate(bd_dim1):
                print((i*j+j))
                print(bd_dim2[0])
                eigvals[:,int(i*j+j)] = bd_dim2[0] 
        print(eigvals)

