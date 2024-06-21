import numpy as np
import glob

files = glob.glob("Hessian_matrix.txt_*",recursive=True)

Hessian = np.zeros((58,58))
for f in files:
    data = np.loadtxt(f)
    indi = [int(d) for d in data[:,1]]
    indj = [int(d) for d in data[:,2]]
    Hessian[indi,indj] = data[:,0]
np.savez("Hessian",Hessian=Hessian)

