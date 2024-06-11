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

class TETB_slim(Calculator):
    def __init__(self,parameters=None,output="TETB_output",cutoff=10,kmesh=(11,11,1)):
        if parameters is None:
            self.rebo_params = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,
                                     40.609282094464604,4.62769509546907,0.7945927858501145,2.2242248220983427])
            self.kc_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                                        1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
            """self.interlayer_hopping_params = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,
                                                        -0.0024695, 0.0003863, -0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983,
                                                            -0.0046855, 0.0007303,0.0000225, -0.0000393])"""
            self.interlayer_hopping_params = np.array([ 4.62415396e2, -4.22834050e2,  3.22270789e2, -2.02691943e2,
                                                        1.03208453e2, -4.11606173e1,  1.21273048e1, -2.35299265,
                                                        2.24178295e-1,  1.10566332e-3, -3.22265467e3,  2.98387007e3,
                                                        -2.36272805e3,  1.58772585e3, -8.92804822e2,  4.10627608e2,
                                                        -1.48945048e2,  4.00880789e1, -7.13860895,  6.32335807e-1])
            """self.intralayer_hopping_params = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519,
                                                        -0.0004838, -0.0000906,-0.3793837, 0.3204470, -0.1956799, 0.0883986,-0.0300733, 0.0074465,
                                                        -0.0008563, -0.0004453, 0.0003842, -0.0001855])"""
            self.intralayer_hopping_params = np.array([-4.57851739,  4.59235008, -4.27957960,  3.16018980,
                                                        -1.47269151,  5.53506664e-2,  5.35176772e-1, -4.55102674e-1,
                                                        1.90353133e-1, -3.61357631e-2,  3.21965395e-1, -3.20369211e-1,
                                                        3.07308402e-1, -2.73762090e-1,  2.19274986e-1, -1.52570366e-1,
                                                        8.31541600e-2, -2.69722311e-2,  2.66753556e-4,  2.31876604e-3])
            self.parameters = np.append(self.rebo_params,self.kc_params)
            self.parameters = np.append(self.parameters,self.interlayer_hopping_params)
            self.parameters = np.append(self.parameters,self.intralayer_hopping_params)
        else:
            self.parameters = parameters
            self.rebo_params = parameters[:9]
            self.kc_params = parameters[9:18]
            self.interlayer_hopping_params = parameters[18:38]
            self.intralayer_hopping_params = parameters[38:]
        self.kpoints_reduced = k_uniform_mesh(kmesh)
        self.nkp = np.shape(self.kpoints_reduced)[0]
        self.cutoff = cutoff
        self.ang_to_bohr =  1.0/.529177
        
        #organize residual potential output
        self.output = output
        self.repo_root = os.path.join("/".join(TETB_GRAPHENE.__file__.split("/")[:-1]),"parameters")
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.rebo_file = "CH_pz.rebo"
        self.rebo_file+="_nkp"+str(self.nkp)
        subprocess.call("cp "+os.path.join(self.repo_root,self.rebo_file)+" "+self.output,shell=True)

        self.kc_file = "KC_insp_pz.txt"
        self.kc_file+="_nkp"+str(self.nkp)
        cwd = os.getcwd()
        os.chdir(self.output)
        write_kcinsp(self.kc_params,self.kc_file)
        write_rebo(self.rebo_params,self.rebo_file)
        os.chdir(cwd)


    def get_tb_energy(self):
        tb_energy = 0
        nocc = self.natoms//2
        intra_hoppings,inter_hoppings =  self.get_hoppings()
        recip_cell = get_recip_cell(self.cell)
        self.kpoints = self.kpoints_reduced @ recip_cell.T

        self.nkp = np.shape(self.kpoints)[0]
        for i in range(self.nkp):
            ham = np.zeros((self.natoms,self.natoms),dtype=np.complex64)
            intra_phases = np.exp((1.0j)*np.dot(self.kpoints[i,:],self.intra_disp.T))
            inter_phases = np.exp((1.0j)*np.dot(self.kpoints[i,:],self.inter_disp.T))
            ham[self.intra_indi,self.intra_indj] += intra_hoppings * intra_phases
            ham[self.intra_indj,self.intra_indi] += np.conj(intra_hoppings*intra_phases)
            ham[self.inter_indi,self.inter_indj] += inter_hoppings * inter_phases
            ham[self.inter_indj,self.inter_indi] += np.conj(inter_hoppings*inter_phases)

            eigvals,_ = np.linalg.eigh(ham)
            tb_energy += 2 * np.sum(eigvals[:nocc])
        return tb_energy/self.nkp

    def get_residual_energy(self,atoms):
        cwd = os.getcwd()
        os.chdir(self.output)
        forces,residual_pe,residual_energy = run_lammps(atoms,self.kc_file,self.rebo_file)
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
        self.get_disp(atoms)

        return self.get_tb_energy() #+ self.get_residual_energy(atoms)


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


if __name__ == "__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    total_energy = False
    bands = True
    sep = 3.35
    a = 2.46
    n=5
    
    atoms=fg.shift.make_graphene(stacking=["A","B"],cell_type='rect',
                            n_layer=2,n_1=n,n_2=n,lat_con=a,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)
    calc = TETB_slim()

    if total_energy:
        energy = calc.get_total_energy(atoms)

    if bands:
        theta=21.78
        p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
        atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=20)

        Gamma = [0,   0,   0]
        K = [2/3,1/3,0]
        Kprime = [1/3,2/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=60
        (kvec,k_dist, k_node) = k_path(sym_pts,nk)

        erange = 5
        evals = calc.get_band_structure(atoms,kvec)
        fig, ax = plt.subplots()
        label=(r'$K$',r'$\Gamma $', r'$M$',r'$K$')
        # specify horizontal axis details
        # set range of horizontal axis
        ax.set_xlim(k_node[0],k_node[-1])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        # add vertical lines at node positions
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        # put title
        ax.set_title("21.78 twist graphene")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy")
        
        nbands = np.shape(evals)[0]
        efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
        fermi_ind = (nbands)//2

        for n in range(np.shape(evals)[0]):
            ax.plot(k_dist,evals[n,:]-efermi,color="black")
            
        # make an PDF figure of a plot
        fig.tight_layout()
        #ax.set_ylim(-erange,erange)
        fig.savefig("theta_21_78_graphene.png")
        plt.clf()
        
    