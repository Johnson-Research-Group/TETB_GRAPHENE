import numpy as np
import scipy
from scipy.optimize import curve_fit
import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import random

def sk_hopping(disp,sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,
               pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10):
    """pairwise Slater Koster Interlayer hopping parameters for pz orbitals of carbon as parameterized by Popov, Van Alsenoy in
     "Low-frequency phonons of few-layer graphene within a tight-binding model". function is fully vectorized

    :param dR: (np.ndarray [N,3]) displacement between 2 atoms [bohr]

    :returns: (np.ndarray [N,]) Hamiltonian matrix elements [eV]
    """
    dR = disp * 1.0/.529177  
    Cpp_sigma = np.array([sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10])
    Cpp_pi = np.array([pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10])
    dRn = np.linalg.norm(dR, axis=1)
    dRn = dR / dRn[:,np.newaxis]
    eV_per_hart=27.2114

    l = dRn[:, 0]
    m = dRn[:, 1]
    n = dRn[:, 2]
    r = np.linalg.norm(dR,axis=1)
    r = np.clip(r, 1, 10)
    aa = 1.0  # [Bohr radii]
    b = rcut #10.0  # [Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)
    Vpp_sigma =  np.polynomial.chebyshev.chebval(y, Cpp_sigma) 
    Vpp_pi =  np.polynomial.chebyshev.chebval(y, Cpp_pi) 


    Vpp_sigma -= Cpp_sigma[0] / 2
    Vpp_pi -= Cpp_pi[0] / 2
    Ezz = n**2 * Vpp_sigma + (1 - n**2) * Vpp_pi
    valmat = Ezz
    return valmat *eV_per_hart

def hopping_training_data(hopping_type="interlayer"):
    data = []
    # flist = subprocess.Popen(["ls", dataset],
    #                       stdout=subprocess.PIPE).communicate()[0]
    # flist = flist.decode('utf-8').split("\n")[:-1]
    # flist = [dataset+x for x in flist]
    flist = glob.glob('../data/hoppings/*.hdf5',recursive=True)
    eV_per_hart=27.2114
    hoppings = np.zeros((1,1))
    disp_array = np.zeros((1,3))
    for f in flist:
        if ".hdf5" in f:
            with h5py.File(f, 'r') as hdf:
                # Unpack hdf
                lattice_vectors = np.array(hdf['lattice_vectors'][:]) #* 1.88973
                atomic_basis =    np.array(hdf['atomic_basis'][:])    #* 1.88973
                tb_hamiltonian = hdf['tb_hamiltonian']
                tij = np.array(tb_hamiltonian['tij'][:]) #* eV_per_hart
                di  = np.array(tb_hamiltonian['displacementi'][:])
                dj  = np.array(tb_hamiltonian['displacementj'][:])
                ai  = np.array(tb_hamiltonian['atomi'][:])
                aj  = np.array(tb_hamiltonian['atomj'][:])
                displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                          dj[:, np.newaxis] * lattice_vectors[1] +\
                          atomic_basis[aj] - atomic_basis[ai]
                
            hoppings = np.append(hoppings,tij)
            disp_array = np.vstack((disp_array,displacement_vector)) 
    hoppings = hoppings[1:]
    disp_array = disp_array[1:,:]
    if hopping_type=="interlayer":
        type_ind = np.where(disp_array[:,2] > 1) # Inter-layer hoppings only, allows for buckling
    else:
        type_ind = np.where(disp_array[:,2] < 1)
    return hoppings[type_ind],disp_array[type_ind]

def fit_hoppings(dft_hoppings,disp_array):
    Cpp_sigma=np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,
                           -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                             -0.0024695, 0.0003863]),
    Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,
    -0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])

    init_params = np.append(Cpp_sigma,Cpp_pi)
    popt,pcov = curve_fit(sk_hopping,disp_array,dft_hoppings,p0=init_params)
    return popt

if __name__=="__main__":
    # fit interlayer parameters
    rcut = 10
    interlayer_hoppings,interlayer_disp = hopping_training_data(hopping_type="interlayer")
    interlayer_params = fit_hoppings(interlayer_hoppings,interlayer_disp)
    print(interlayer_params)
    Cpp_sigma_interlayer = interlayer_params[:10]
    Cpp_pi_interlayer = interlayer_params[10:]
    #interlayer_params = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863, -0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])
    [sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10] = interlayer_params[:10]
    [pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10] = interlayer_params[10:]
    interlayer_fit_hoppings = sk_hopping(interlayer_disp,sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,
               pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10)

    plt.scatter(np.linalg.norm(interlayer_disp,axis=1) / .529177,interlayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(interlayer_disp,axis=1)/ .529177,interlayer_fit_hoppings,label="SK")
    plt.xlabel("distance (bohr)")
    plt.ylabel("hoppings (eV)")
    plt.legend()
    plt.title("interlayer hoppings fit")
    plt.savefig("interlayer_hoppings.png")
    plt.clf()

    # fit intralayer parameters
    rcut = 7
    intralayer_hoppings,intralayer_disp = hopping_training_data(hopping_type="intralayer")
    intralayer_params = fit_hoppings(intralayer_hoppings,intralayer_disp)
    Cpp_sigma_intralayer = intralayer_params[:10]
    Cpp_pi_intralayer = intralayer_params[10:]
    [sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10] = intralayer_params[:10]
    [pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10] = intralayer_params[10:]
    intralayer_fit_hoppings = sk_hopping(intralayer_disp,sp1,sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,
               pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10)

    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings.png")
    plt.clf()

    nn_dist = 1.42
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist,1.05*nn_dist)
    plt.ylim(-85,-70)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_1nn.png")
    plt.clf()

    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist*np.sqrt(3),1.05*nn_dist*np.sqrt(3))
    plt.ylim(-5,10)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_2nn.png")
    plt.clf()

    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_hoppings,label="DFT")
    plt.scatter(np.linalg.norm(intralayer_disp,axis=1),intralayer_fit_hoppings,label="SK")
    plt.xlabel("distance (angstroms)")
    plt.ylabel("hoppings (eV)")
    plt.xlim(0.95*nn_dist*2,1.05*nn_dist*2)
    plt.ylim(-15,0)
    plt.legend()
    plt.title("intralayer hoppings fit")
    plt.savefig("intralayer_hoppings_3nn.png")
    plt.clf()

    np.savez("tb_params",Cpp_sigma_interlayer=Cpp_sigma_interlayer,Cpp_sigma_intralayer=Cpp_sigma_intralayer,
             Cpp_pi_interlayer=Cpp_pi_interlayer,Cpp_pi_intralayer=Cpp_pi_intralayer,distance_units="angstroms",energy_units="eV")
    

    def fun(x, y):
        return x**2 + y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dr=1e-2
    x = y = np.arange(-1.0, 1.0, 0.05)*dr
    X, Y = np.meshgrid(x, y)
    npoints = len(np.ravel(X))
    zs = np.zeros(npoints)
    for i in range(npoints):
        zs[i] = np.linalg.norm(sk_hopping(intralayer_disp,np.ravel(X)[i]+sp1, sp2,sp3,sp4,sp5,sp6,sp7,sp8,sp9,sp10,
               np.ravel(Y)[i]+pp1,pp2,pp3,pp4,pp5,pp6,pp7,pp8,pp9,pp10)-intralayer_hoppings)/len(intralayer_hoppings)
    Z = zs.reshape(X.shape)
    Z -= np.min(Z)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel(r'$\theta=$sp1')
    ax.set_ylabel(r'$\theta=p \pi 1$')
    ax.set_zlabel('Cost')
    plt.title('Hopping Cost surface')
    plt.savefig("cost_surface.png")
    

