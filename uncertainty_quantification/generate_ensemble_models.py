import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import os
from ase.lattice.hexagonal import Graphite
from reformat_TETB_GRAPHENE_calc import TETB_GRAPHENE_Calc
from TB_parameters_v2 import *
import h5py
import glob
import ase.db

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

def calc_Hessian(init_param_dict,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data,dt=1e-2):
    init_param = np.append(init_param_dict["intralayer potential"],init_param_dict["interlayer potential"])
    init_param = np.append(init_param,init_param_dict["interlayer hoppings"])
    init_param = np.append(init_param,init_param_dict["intralayer hoppings"])
    n_params = len(init_param)
    Hessian = np.zeros((n_params,n_params))
    for i in range(n_params):
        for j in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = 1
            ej = np.zeros(n_params)
            ej[j] = 1
            f1 = get_Cost(init_param + dt * ei + dt * ej,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
            f2 = get_Cost(init_param + dt * ei - dt * ej,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
            f3 = get_Cost(init_param - dt * ei + dt * ej,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
            f4 = get_Cost(init_param - dt * ei - dt * ej,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
            Hessian[i,j] = (f1-f2-f3+f4)/(4*dt*dt)
    return Hessian

def calc_energy_error(calc,parameters,interlayer_energy_db,intralayer_energy_db,interlayer_hopping_data,intralayer_hopping_data):
    intralayer_params = parameters[:9]
    interlayer_params = parameters[9:18]
    interlayer_hopping_params = parameters[18:38]
    intralayer_hopping_params = parameters[38:]
    energy_rmse_array = []
    hopping_rmse_array=[]
    for row in interlayer_energy_db.select():
        atoms = interlayer_energy_db.get_atoms(id = row.id)
        pos = atoms.positions
        mean_z = np.mean(pos[:,2])
        top_ind = np.where(pos[:,2]>mean_z)
        mol_id = np.ones(len(atoms),dtype=np.int64)
        mol_id[top_ind] = 2
        atoms.set_array("mol-id",mol_id)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energy_rmse_array = np.linalg.norm((energy-row.data.total_energy)/len(atoms))/row.data.total_energy

    for row in intralayer_energy_db.select():
        atoms = intralayer_energy_db.get_atoms(id = row.id)
        atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energy_rmse_array = np.append(energy_rmse_array,np.linalg.norm((energy-row.data.total_energy)/len(atoms))/row.data.total_energy)
    
    hoppings = popov_hopping(interlayer_hopping_data["disp"],params=np.vstack((interlayer_hopping_params[:10],interlayer_hopping_params[10:])))
    hopping_rmse_array = np.append(hopping_rmse_array,np.linalg.norm((hoppings-interlayer_hopping_data["hopping"]))/interlayer_hopping_data["hopping"])

    hoppings = porezag_hopping(intralayer_hopping_data["disp"],params=np.vstack((intralayer_hopping_params[:10],intralayer_hopping_params[10:])))
    hopping_rmse_array = np.append(hopping_rmse_array,np.linalg.norm((hoppings-intralayer_hopping_data["hopping"]))/intralayer_hopping_data["hopping"])


    return np.array(energy_rmse_array),np.array(hopping_rmse_array)

def get_Cost(parameters,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data,a=0.5,b=0.5,kmesh = (11,11,1)):
    intralayer_params = parameters[:9]
    interlayer_params = parameters[9:18]
    interlayer_hopping_params = parameters[18:38]
    intralayer_hopping_params = parameters[38:]
    model_dict = dict({"tight binding parameters":{
                                                        "interlayer":{"hopping":{"model":"popov","params":np.vstack((interlayer_hopping_params[:10],interlayer_hopping_params[10:]))},
                                                                     "overlap":{"model":"popov","params":None}},
                                                        "intralayer":{"hopping":{"model":"porezag","params":np.vstack((intralayer_hopping_params[:10],intralayer_hopping_params[10:]))}},
                                                              "overlap":{"model":"porezag","params":None}
                                                              },
                        "basis":"pz",
                        "kmesh":kmesh,
                        "parallel":"joblib",
                        "intralayer potential":intralayer_params,
                        "interlayer potential":interlayer_params,
                        'output':"ensemble_cost_output"})
    calc_obj = TETB_GRAPHENE_Calc(model_dict,use_overlap=False)
    total_energy_error,hopping_error = calc_energy_error(calc_obj,parameters,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
    cost = a*np.mean(total_energy_error) + b*np.mean(hopping_error)
    return cost

def generate_Ensemble(init_param_dict,Hessian,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data,Nensembles=500,R=1):
    """init_param are params of model that minimize the cost """
    init_param = np.append(init_param_dict["intralayer potential"],init_param_dict["interlayer potential"])
    init_param = np.append(init_param,init_param_dict["interlayer hoppings"])
    init_param = np.append(init_param,init_param_dict["intralayer hoppings"])
    n_params = len(init_param)
    ensemble_parameters = np.zeros((n_params,Nensembles))
    cost_array = np.zeros(Nensembles)
    ensemble_parameters[:,0] = init_param
    cost_array[0] = get_Cost(init_param,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data) 
    T_0 = 2*cost_array[0]/n_params
    Hessian_eigvals, Hessian_eigvecs = np.linalg.eig(Hessian)
    n_accept=1
    while n_accept < Nensembles:
        prev_cost = cost_array[n_accept-1]
        d_t = np.zeros(n_params)
        for i in range(n_params):
            for j in range(n_params):
                d_t[i] += np.real(np.sqrt(R/np.max(Hessian_eigvals[j],1)) * Hessian_eigvecs[i,j] * np.random.normal())
        
        cost_t_next = get_Cost(ensemble_parameters[:,0] + d_t,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data) #n_accept
        accept_prob = np.exp(-(cost_t_next - prev_cost)/T_0)
        if cost_t_next < prev_cost or np.random.rand() < accept_prob:
            ensemble_parameters[:,n_accept+1] = ensemble_parameters[:,n_accept] + d_t
            n_accept +=1

    return ensemble_parameters

if __name__=="__main__":

    nkp = 121

    hopping_params = np.load("tb_params.npz")
    intralayer_potential = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,
                            14522.273379352482,7.855493960028371,40.609282094464604,
                            4.62769509546907,0.7945927858501145,2.2242248220983427])
    interlayer_potential = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                             1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
    interlayer_hoppings = np.append(hopping_params["Cpp_sigma_interlayer"],hopping_params["Cpp_pi_interlayer"])
    intralayer_hoppings = np.append(hopping_params["Cpp_sigma_intralayer"],hopping_params["Cpp_pi_intralayer"])
    interlayer_db =  ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
    intralayer_db = db = ase.db.connect('../data/monolayer_nkp'+str(nkp)+'.db')
    interlayer_hopping_data = hopping_training_data(hopping_type="interlayer")
    intralayer_hopping_data = hopping_training_data(hopping_type="intralayer")

    init_params={}
    init_params.update({"intralayer potential":intralayer_potential})
    init_params.update({"interlayer potential":interlayer_potential})
    init_params.update({"interlayer hoppings":interlayer_hoppings})
    init_params.update({"intralayer hoppings":intralayer_hoppings})
    if not os.path.exists("hessian.npz"):
        Hessian = calc_Hessian(init_params,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data)
        np.savez("hessian",Hessian=Hessian)
    else:
        Hessian = np.load("hessian.npz")["Hessian"]
    
    ensemble_parameters = generate_Ensemble(init_params,Hessian,interlayer_db,intralayer_db,interlayer_hopping_data,intralayer_hopping_data,Nensembles=500,R=1)
    np.savez("ensemble_parameters",ensemble_parameters=ensemble_parameters)
    

