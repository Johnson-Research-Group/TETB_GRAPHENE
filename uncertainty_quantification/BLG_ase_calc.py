import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
import TETB_GRAPHENE
import scipy.linalg as spla
from scipy.spatial.distance import cdist
from TB_parameters_v2 import *
from TB_Utils import *
from Lammps_Utils import *
from time import time

class BLG_classical(Calculator):
    def __init__(self,parameters=None,output="blg_output",cutoff=10):
        if parameters is None:
            self.rebo_params = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
            self.kc_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                        0.719345289329483, 3.293082477932360, 13.906782892134125])
            
            self.parameters = np.append(self.rebo_params,self.kc_params)
        else:
            self.parameters = parameters
            self.rebo_params = parameters[:9]
            self.kc_params = parameters[9:]

        self.cutoff = cutoff
        self.ang_to_bohr =  1.0/.529177
        
        #organize residual potential output
        self.output = output
        self.repo_root = os.path.join("/".join(TETB_GRAPHENE.__file__.split("/")[:-1]),"parameters")
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.rebo_file = "CH.rebo"
        subprocess.call("cp "+os.path.join(self.repo_root,self.rebo_file)+" "+self.output,shell=True)

        self.kc_file = "KC.txt"
        cwd = os.getcwd()
        os.chdir(self.output)
        write_kc(self.kc_params,self.kc_file)
        write_rebo(self.rebo_params,self.rebo_file)
        os.chdir(cwd)


    def get_residual_energy(self,atoms):
        cwd = os.getcwd()
        os.chdir(self.output)
        forces,residual_pe,residual_energy = run_lammps(atoms,self.kc_file,self.rebo_file,type="classical")
        os.chdir(cwd)
        return residual_pe
    
    def get_band_structure(self,atoms,kpoints):

        self.get_disp(atoms)

        intra_hoppings,inter_hoppings =  self.get_hoppings()
        recip_cell = get_recip_cell(self.cell)
        self.kpoint_path = kpoints @ recip_cell.T
        self.nkp = np.shape(self.kpoint_path)[0]
        eigvals_k = np.zeros((self.natoms,self.nkp))
        for i in range(self.nkp):
            ham = np.zeros((self.natoms,self.natoms),dtype=np.complex64)
            intra_phases = np.exp((1.0j)*np.dot(self.kpoint_path[i,:],self.intra_disp.T))
            inter_phases = np.exp((1.0j)*np.dot(self.kpoint_path[i,:],self.inter_disp.T))
            ham[self.intra_indi,self.intra_indj] += intra_hoppings * intra_phases
            ham[self.intra_indj,self.intra_indi] += np.conj(intra_hoppings*intra_phases)
            ham[self.inter_indi,self.inter_indj] += inter_hoppings * inter_phases
            ham[self.inter_indj,self.inter_indi] += np.conj(inter_hoppings*inter_phases)

            eigvals,_ = np.linalg.eigh(ham)
            eigvals_k[:,i] = eigvals
        return eigvals_k
    
    def get_total_energy(self,atoms):
        if not atoms.has("mol-id"):
            pos = atoms.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            mol_id = np.ones(len(atoms),dtype=np.int64)
            mol_id[top_ind] = 2
            atoms.set_array("mol-id",mol_id)

        return  self.get_residual_energy(atoms)


    ##############################################################################################

    # Utils
    
    ##############################################################################################

    def get_hoppings(self):
        #intralayer
        intra_hoppings = porezag_hopping(self.intra_disp,params=np.vstack((self.intralayer_hopping_params[:10],self.intralayer_hopping_params[10:])))/2  # Divide by 2 since we are double counting every pair

        #interlayer
        inter_hoppings = popov_hopping(self.inter_disp,params=np.vstack((self.interlayer_hopping_params[:10],self.interlayer_hopping_params[10:])))/2  # Divide by 2 since we are double counting every pair

        return intra_hoppings,inter_hoppings
    
    def get_disp(self,atoms):

        self.positions = atoms.positions*self.ang_to_bohr
        self.atom_types = atoms.get_array("mol-id")
        self.natoms = len(self.atom_types)
        self.cell = atoms.get_cell()*self.ang_to_bohr

        diFull = []
        djFull = []
        extended_coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                extended_coords += list(self.positions[:, :] + self.cell[0, np.newaxis] * dx + self.cell[1, np.newaxis] * dy)
                diFull += [dx] * self.natoms
                djFull += [dy] * self.natoms
        distances = cdist(self.positions, extended_coords)

        i, j = np.where((distances > 0.1)  & (distances < self.cutoff))
        di = np.array(diFull)[j]
        dj = np.array(djFull)[j]
        self.i  = np.array(i)
        self.j  = np.array(j % self.natoms)

        intra_valid_indices = self.atom_types[self.i] == self.atom_types[self.j]
        self.intra_indi = self.i[intra_valid_indices]
        self.intra_indj = self.j[intra_valid_indices]

        inter_valid_indices = self.atom_types[self.i] != self.atom_types[self.j]
        self.inter_indi = self.i[inter_valid_indices]
        self.inter_indj = self.j[inter_valid_indices]

        self.inter_disp = di[inter_valid_indices, np.newaxis] * self.cell[0] +\
                            dj[inter_valid_indices, np.newaxis] * self.cell[1] +\
                            self.positions[self.inter_indj] - self.positions[self.inter_indi]
        
        self.intra_disp = di[intra_valid_indices, np.newaxis] * self.cell[0] +\
                            dj[intra_valid_indices, np.newaxis] * self.cell[1] +\
                            self.positions[self.intra_indj] - self.positions[self.intra_indi]