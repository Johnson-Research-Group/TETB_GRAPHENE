import numpy as np
import matplotlib.pyplot as plt

def hop(r):
    #return -np.log(r)
    return -np.exp(-r)

def Ham_fxn(r):
    n=2
    Ham = np.zeros((n,n))
    for i in range(n-1):
        Ham[i,i+1]= hop(r)
        Ham[i+1,i] = hop(r)
    return Ham
if __name__=="__main__":
    nsteps = 10
    energies = np.zeros(nsteps)
    r = np.linspace(1,7,10)
    for i in range(nsteps):
        ham = Ham_fxn(r[i])
        evals,evecs = np.linalg.eig(ham)
        nocc = np.shape(evals)[0]//2
        energies[i]+=np.sum(np.sort(evals.real)[:nocc])
    plt.plot(r,energies-energies[-1],label="energy")
    plt.plot(r,hop(r),label="hop")
    plt.plot(r,np.gradient(hop(r)),label="grad")
    plt.legend()
    plt.savefig("hop_energy.png")
    plt.clf()