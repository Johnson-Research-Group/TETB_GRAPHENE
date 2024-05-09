import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import os
from ase.lattice.hexagonal import Graphite
import reformat_TETB_GRAPHENE.TETB_GRAPHENE_calc

def calc_Hessian(init_param,dt=1e-2):
    n_params = len(init_param)
    Hessian = np.zeros((n_params,n_params))
    for i in range(n_params):
        for j in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = 1
            ej = np.zeros(n_params)
            ej[j] = 1
            f1 = get_Cost(init_param + dt * ei + dt * ej)
            f2 = get_Cost(init_param + dt * ei - dt * ej)
            f3 = get_Cost(init_param - dt * ei + dt * ej)
            f4 = get_Cost(init_param - dt * ei - dt * ej)
            Hessian[i,j] = (f1-f2-f3+f4)/(4*dt*dt)
    return Hessian

def calc_energy_error(calc,total_energy_db,hopping_db):
    energy_rmse_array = []
    hopping_rmse_array=[]
    for row in total_energy_db.select():
        atoms = total_energy_db.get_atoms(id = row.id)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        hoppings = atoms.get_hoppings()
        energy_rmse_array.append(np.linalg.norm((energy-row.data.total_energy)/len(atoms))/row.data.total_energy)
    

    for row in hopping_db.select():
        atoms = hopping_db.get_atoms(id = row.id)
        atoms.calc = calc
        hoppings = atoms.get_hoppings()
        hopping_rmse_array.append(np.linalg.norm((hoppings-row.data.hoppings))/row.data.hoppings)
    return np.array(energy_rmse_array),np.array(hopping_rmse_array)

def get_Cost(parameters,db,kmesh = (11,11,1)):
    intralayer_params = parameters["intralayer potential"]
    interlayer_params = parameters["interlayer potential"]
    interlayer_hopping_params = parameters["interlayer hoppings"]
    intralayer_hopping_params = parameters["intralayer hoppings"]
    model_dict = dict({"tight binding parameters":{"interlayer":{"name":"popov","hopping param":interlayer_hopping_params},
                                                "intralayer":{"name":"porezag","hopping param":intralayer_hopping_params}}, 
                        "basis":"pz",
                        "kmesh":kmesh,
                        "parallel":"joblib",
                        "intralayer potential":intralayer_params,
                        "interlayer potential":interlayer_params,
                        'output':"ensemble_cost_output"})
    calc_obj = reformat_TETB_GRAPHENE.TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
    total_energy_error,hopping_error = calc_energy_error(calc_obj,db)
    cost = np.mean(total_energy_error) + np.mean(hopping_error)
    return cost

def generate_Ensemble(init_param,Hessian,Nensembles=500,R=1):
    """init_param are params of model that minimize the cost """
    n_params = len(init_param)
    ensemble_parameters = np.zeros((n_params,Nensembles))
    cost_array = np.zeros(Nensembles)
    ensemble_parameters[:,0] = init_param
    cost_array[0] = get_Cost(init_param) 
    T_0 = 2*cost_array[0]/n_params
    Hessian_eigvals, Hessian_eigvecs = np.linalg.eig(Hessian)
    n_accept=1
    while n_accept < Nensembles:
        prev_cost = cost_array[n_accept-1]
        d_t = np.zeros(n_params)
        for i in range(n_params):
            for j in range(n_params):
                d_t[i] += np.real(np.sqrt(R/np.max(Hessian_eigvals[j],1)) * Hessian_eigvecs[i,j] * np.random.normal())
        
        cost_t_next = get_Cost(ensemble_parameters[:,n_accept] + d_t)
        accept_prob = np.exp(-(cost_t_next - prev_cost)/T_0)
        if cost_t_next < prev_cost or np.random.rand() < accept_prob:
            ensemble_parameters[:,n_accept+1] = ensemble_parameters[:,n_accept] + d_t
            n_accept +=1

    return ensemble_parameters

if __name__=="__main__":
    init_params={}
    init_params.update({"intralayer potential":intralayer_potential})
    init_params.update({"interlayer potential":interlayer_potential})
    init_params.update({"interlayer hoppings":interlayer_hoppings})
    init_params.update({"intralayer hoppings":interlayer_hoppings})
    if not os.path.exists("hessian.npz"):
        Hessian = calc_Hessian(init_params)
        np.savez("hessian",Hessian=Hessian)
    else:
        Hessian = np.load("hessian.npz")["Hessian"]
    
    ensemble_parameters = generate_Ensemble(init_params,Hessian,Nensembles=500,R=1)
    np.savez("ensemble_parameters",ensemble_parameters=ensemble_parameters)
    

