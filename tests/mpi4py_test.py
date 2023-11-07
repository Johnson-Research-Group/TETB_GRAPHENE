import joblib
from joblib import Parallel, delayed
from mpi4py import MPI
import numpy as np

def tb_fxn(i):
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
        return energy

if __name__=="__main__":
    tsteps = 10
    for i in range(tsteps):
        energy = get_energy()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank ==0:
            print(i,energy)
