from corner import corner
from kliff.calculators import Calculator as Kliff_calc
from kliff.dataset.weight import MagnitudeInverseWeight
from kliff.loss import Loss
from kliff.models.parameter_transform import LogParameterTransform
from kliff.uq import MCMC, get_T0, autocorr, mser, rhat
from kliff.loss import Loss
from TETB_MODEL_KLIFF import TETB_KLIFF_Model
from schwimmbad import MPIPool
from multiprocessing import Pool
import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
from TETB_LOSS import *
from TETB_slim import *
import ase.db
from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.dataset.dataset import Configuration

def hopping_training_data(hopping_type="interlayer"):
    data = []
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
                tij = np.array(tb_hamiltonian['tij'][:]) * eV_per_hart
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
    return {"hopping":hoppings[type_ind],"disp":disp_array[type_ind]}

def create_Dataset(interlayer_db,intralayer_db):
    configs = []

    for i,row in enumerate(interlayer_db.select()):
        atoms = interlayer_db.get_atoms(id = row.id)
        pos = atoms.positions
        mean_z = np.mean(pos[:,2])
        top_ind = np.where(pos[:,2]>mean_z)
        mol_id = np.ones(len(atoms),dtype=np.int64)
        mol_id[top_ind] = 2
        atoms.set_array("mol-id",mol_id)

        a_config = Configuration(cell=atoms.get_cell(),
                    species= atoms.get_chemical_symbols(),
                    coords= atoms.positions,
                    PBC= [True,True,True],
                    energy= row.data.total_energy)
        configs.append(a_config)

    for i,row in enumerate(intralayer_db.select()):
        atoms = intralayer_db.get_atoms(id = row.id)
        atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))

        a_config = Configuration(cell=atoms.get_cell(),
                    species= atoms.get_chemical_symbols(),
                    coords= atoms.positions,
                    PBC= [True,True,True],
                    energy= row.data.total_energy)
        configs.append(a_config)

    return configs

if __name__=="__main__":
    """ run mcmc
    $ export MPIEXEC_OPTIONS="--bind-to core --map-by slot:PE=<num_openmp_processes> port-bindings"
    $ mpiexec -np <num_mpi_workers> ${MPIEXEC_OPTIONS} python script.py
    """
    #define minimal cost model parameters
    rebo_params = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,
                                     40.609282094464604,4.62769509546907,0.7945927858501145,2.2242248220983427])
    kc_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                                1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
    interlayer_hopping_params = np.array([ 4.62415396e2, -4.22834050e2,  3.22270789e2, -2.02691943e2,
                                                1.03208453e2, -4.11606173e1,  1.21273048e1, -2.35299265,
                                                2.24178295e-1,  1.10566332e-3, -3.22265467e3,  2.98387007e3,
                                                -2.36272805e3,  1.58772585e3, -8.92804822e2,  4.10627608e2,
                                                -1.48945048e2,  4.00880789e1, -7.13860895,  6.32335807e-1])
    intralayer_hopping_params = np.array([-4.57851739,  4.59235008, -4.27957960,  3.16018980,
                                                -1.47269151,  5.53506664e-2,  5.35176772e-1, -4.55102674e-1,
                                                1.90353133e-1, -3.61357631e-2,  3.21965395e-1, -3.20369211e-1,
                                                3.07308402e-1, -2.73762090e-1,  2.19274986e-1, -1.52570366e-1,
                                                8.31541600e-2, -2.69722311e-2,  2.66753556e-4,  2.31876604e-3])
    opt_params = {}
    for i in range(len(rebo_params)):
        opt_params.update({"rebo_"+str(i):[[rebo_params[i],"-INF","INF"]]})

    for i in range(len(kc_params)):
        opt_params.update({"kc_"+str(i):[[kc_params[i],"-INF","INF"]]})

    for i in range(len(interlayer_hopping_params)):
        opt_params.update({"interlayer_hopping_params_"+str(i):[[interlayer_hopping_params[i],"-INF","INF"]]})

    for i in range(len(intralayer_hopping_params)):
        opt_params.update({"intralayer_hopping_params_"+str(i):[[intralayer_hopping_params[i],"-INF","INF"]]})

    nkp = 121
    interlayer_db =  ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
    intralayer_db = db = ase.db.connect('../data/monolayer_nkp'+str(nkp)+'.db')
    configs = create_Dataset(interlayer_db,intralayer_db)

    interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
    intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")
    p0 = np.append(rebo_params,kc_params)
    p0= np.append(p0,interlayer_hopping_params)
    p0 = np.append(0,intralayer_hopping_params)

    model = TETB_KLIFF_Model()

    model.set_opt_params(**opt_params)

    calculator = Kliff_calc(model)
    ca = calculator.create(configs,use_forces=False)
    hopping_data = {"interlayer":interlayer_hopping_data,"intralayer":intralayer_hopping_data}
    loss = LossTETBModel(
        calculator,
        nprocs=1,
        hopping_data = hopping_data
    )

    #define hyperparameters
    ndim = calculator.get_num_opt_params()
    bounds = np.tile([-8.0, 8.0], (ndim, 1))
    iterations = 1 #50
    nwalkers = 2 * ndim
    T0 = get_T0(loss)
    ntemps = 1

    #run monte carlo sampling
    sampler = MCMC(
        loss, nwalkers,  ntemps = ntemps ,logprior_args=(bounds,),
    ) #logprior_fn, logprior_args,

    
    p0 = np.empty((ntemps, nwalkers, ndim))
    for ii, bound in enumerate(bounds):
        p0[:, :, ii] = np.random.uniform(*bound, (ntemps, nwalkers))
    #sampler.pool = MPIPool()
    samples.pool = Pool(60)
    sampler.run_mcmc(p0, iterations)
    sampler.pool.close() 

    # Retrieve the chain
    chain = sampler.chain

    #determine equilibration time
    mser_array = np.empty((ntemps, nwalkers, ndim))
    for tidx in range(ntemps):
        for widx in range(nwalkers):
            for pidx in range(ndim):
                mser_array[tidx, widx, pidx] = mser(
                    chain[tidx, widx, :, pidx], dmin=0, dstep=10, dmax=-1
                )

    burnin = int(np.max(mser_array))
    print(f"Estimated burn-in time: {burnin}")

    # Estimate the autocorrelation length for each temperature
    chain_no_burnin = chain[:, :, burnin:]

    acorr_array = np.empty((ntemps, nwalkers, ndim))
    for tidx in range(ntemps):
        acorr_array[tidx] = autocorr(chain_no_burnin[tidx], c=1, quiet=True)

    thin = int(np.ceil(np.max(acorr_array)))
    print(f"Estimated autocorrelation length: {thin}")

    # Assess the convergence for each temperature
    samples = chain_no_burnin[:, :, ::thin]

    threshold = 1.1  # Threshold for rhat
    rhat_array = np.empty(ntemps)
    for tidx in range(ntemps):
        rhat_array[tidx] = rhat(samples[tidx])

    print(f"$\hat{{r}}^p$ values: {rhat_array}")



    
