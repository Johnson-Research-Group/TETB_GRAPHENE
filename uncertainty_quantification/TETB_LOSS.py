import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.optimize
from loguru import logger
from TB_Utils import *
from kliff import parallel
from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.calculators.calculator_torch import CalculatorTorch
from kliff.dataset.weight import Weight
from kliff.error import report_import_error
from kliff.dataset.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.models.parameter_transform import ParameterTransform
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress
from kliff.loss import Loss
from ase import Atoms
from datetime import datetime
from TETB_slim import TETB_slim
import subprocess
import time
import ase.io

class LossTETBModel:
    """
    Loss function class to optimize the physics-based potential parameters.

    Args:
        calculator: Calculator to compute prediction from atomic configuration using
            a potential model.
        nprocs: Number of processes to use..
        residual_fn: function to compute residual, e.g. :meth:`energy_forces_residual`,
            :meth:`energy_residual`, and :meth:`forces_residual`. See the documentation
            of :meth:`energy_forces_residual` for the signature of the function.
            Default to :meth:`energy_forces_residual`.
        residual_data: data passed to ``residual_fn``; can be used to fine tune the
            residual function. Default to
            {
                "normalize_by_natoms": True,
            }
            See the documentation of :meth:`energy_forces_residual` for more.
    """
    def __init__(self,atoms,hopping_data:Optional[Dict[str,Any]] = None, data_file_dir = "TETB_data_files", opt_params=None):
        self.data_file_dir = os.path.join(os.getcwd(),data_file_dir)
        if not os.path.exists(self.data_file_dir):
            os.mkdir(self.data_file_dir)
            self.write_data_files(atoms)
        self.atoms = atoms
        self.interlayer_hopping_data = hopping_data["interlayer"]
        self.intralayer_hopping_data = hopping_data["intralayer"]
        self.ref_energy = []
        for a in atoms:
            self.ref_energy.append(a.total_energy)
        self.ref_energy = np.array(self.ref_energy)
        self.calculator = TETBcalculator(opt_params)

    def write_data_files(self,atoms):
        for i,a in enumerate(atoms):
            ase.io.write(os.path.join(self.data_file_dir,"tegt.data_"+str(i+1)),a,format="lammps-data",atom_style = "full")

    def get_tb_energies(self,calc):
        tb_energy = []
        for i,a in enumerate(self.atoms):
            tmp_energy = calc.get_tb_energy(a)
            tb_energy.append(tmp_energy)
        return np.array(tb_energy)

    def _get_loss(self, x):
        start = time.time()
        #only write parameter files once per set of parameters
        #calculate tb energy and residual energy
        self.output = "TETB_output_"+ str(hash(datetime.now()) )
        param_list = np.array([np.squeeze(x[p].value) for p in x])[1:]
        calc = TETB_slim(parameters=param_list,output = self.output)
        tb_energies = self.get_tb_energies(calc)
        #calculate residual energy of all structures stored in self.data_file_dir
        lammps_energies = calc.get_residual_energy(self.data_file_dir)
        subprocess.call("rm -rf "+self.output,shell=True)

        predicted_energies = tb_energies + lammps_energies
        residual = (predicted_energies - self.ref_energy)/self.ref_energy

        energy_loss = 0.5 * np.linalg.norm(residual) ** 2
        
        #these evaluations of hopping parameterizations are very quick
        interlayer_hopping_params = x[18:38]
        intralayer_hopping_params = x[38:]
        hoppings = popov_hopping(self.interlayer_hopping_data["disp"],params=np.vstack((interlayer_hopping_params[:10],interlayer_hopping_params[10:])))
        hopping_rmse_array = np.linalg.norm((hoppings-self.interlayer_hopping_data["hopping"]))/self.interlayer_hopping_data["hopping"]

        hoppings = porezag_hopping(self.intralayer_hopping_data["disp"],params=np.vstack((intralayer_hopping_params[:10],intralayer_hopping_params[10:])))
        hopping_rmse_array = np.append(hopping_rmse_array,np.linalg.norm((hoppings-self.intralayer_hopping_data["hopping"]))/self.intralayer_hopping_data["hopping"])

        loss = 0.5 * energy_loss + 0.5 * np.linalg.norm(hopping_rmse_array)
        end = time.time()
        print("time for loss function = ",end-start)
        return loss
    
class TETBcalculator:
    def __init__(self,params=None):
        self.species=None
        if params is None:
            self.rebo_params = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,
                                    40.609282094464604,4.62769509546907,0.7945927858501145,2.2242248220983427])
            self.kc_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                                    1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
            self.interlayer_hopping_params = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,
                                                    -0.0024695, 0.0003863, -0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983,
                                                        -0.0046855, 0.0007303,0.0000225, -0.0000393])
            self.intralayer_hopping_params = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519,
                                                        -0.0004838, -0.0000906,-0.3793837, 0.3204470, -0.1956799, 0.0883986,-0.0300733, 0.0074465,
                                                        -0.0008563, -0.0004453, 0.0003842, -0.0001855])
        else:
            self.rebo_params = params[:9]
            self.kc_params = params[9:18]
            self.interlayer_hopping_params = params[18:38]
            self.intralayer_hopping_params = params[38:]
        self.init_model_params()

    def init_model_params(self):
        n = self._get_num_params()
        self.model_params = {"cutoff": Parameter(value=[10.0*0.529 for _ in range(n)])}
        
        for i in range(len(self.rebo_params)):
            self.model_params.update({"rebo_"+str(i):Parameter(value=[self.rebo_params[i] for _ in range(n)])})

        for i in range(len(self.kc_params)):
            self.model_params.update({"kc_"+str(i):Parameter(value=[self.kc_params[i] for _ in range(n)])})

        for i in range(len(self.interlayer_hopping_params)):
            self.model_params.update({"interlayer_hopping_params_"+str(i):Parameter(value=[self.interlayer_hopping_params[i] for _ in range(n)])})

        for i in range(len(self.intralayer_hopping_params)):
            self.model_params.update({"intralayer_hopping_params_"+str(i):Parameter(value=[self.intralayer_hopping_params[i] for _ in range(n)])})
        

        return self.model_params

    def _get_num_params(self):
        if self.species is None:
            n = 1
        else:
            n = len(self.species)

        return (n + 1) * n // 2
    
    def get_opt_params(self):
        return self.model_params
    
    def get_num_opt_params(self):
        return len(self.model_params)
    
    def get_opt_params_bounds(self):
        bounds = []
        for idx in self._index:
            name = idx.name
            c_idx = idx.c_idx
            lower = self.model_params[name].lower_bound[c_idx]
            upper = self.model_params[name].upper_bound[c_idx]
            bounds.append([lower, upper])

        return bounds