# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:28:32 2023

@author: danpa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:14:50 2023

@author: danpa
"""
import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
import joblib
from joblib import Parallel, delayed
import TETB_GRAPHENE
import scipy.linalg as spla
import dask
from dask.distributed import Client, LocalCluster
from scipy.spatial.distance import cdist
from TB_parameters_v2 import *
from time import time
#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces in parallel

class TETB_GRAPHENE_Calc(Calculator):
    
    implemented_properties = ['energy','forces','potential_energy']
    def __init__(self,model_dict=None,restart_file=None,use_overlap=True, **kwargs):
        """
        ase calculator object that calculates total energies and forces given a 
        total energy tight binding model. can set kmesh to be either (1,1,1), (11,11,1) or (15,15,1).
        Parameters:
        :param model_dict: (dict) dictionary containing info on tb model, number of kpoints, interatomic corrective potentials
                            defaults: {"tight binding parameters":None, (str) tight binding model
                                        "intralayer potential":None, (str) can be path to REBO potential file or keyword
                                        "interlayer potential":None, (str) can be path to interlayer potential file or keyword
                                        "kmesh":(1,1,1), (tuple) mesh of kpoints
                                        "parallel":serial, (str) type of parallelization to use in tb calculation
                                        "output":".", (str) optional output directory
                                        } 
                            tight binding parameters keywords = {None,"popov"} **will default to classical potentials in tight binding parameters == None
                            parallel keywords = {"dask","joblib","serial"} **Note only dask can handle parallelization over multiple compute nodes
                            intralayer potential keywords = {REBO, Pz rebo} 
                            interlayer potential keywords = {kolmogorov crespi, Pz KC inspired} 
        :param restart_file: (str) filename for restart file. default: cwd
        create_model() will automatically select correct potential given the input tight binding model and number of kpoints
        """
        Calculator.__init__(self, **kwargs)
        self.model_dict=model_dict

        self.repo_root = os.path.join("/".join(TETB_GRAPHENE.__file__.split("/")[:-1]))
        #self.repo_root = os.getcwd()
        self.param_root = os.path.join(self.repo_root,"parameters")
        self.option_to_file={
                     "Rebo":"CH.rebo",
                     "Pz pairwise":"pz_pairwise_correction.table",
                     "Pz rebo":"CH_pz.rebo",
                     "Pz rebo nkp225":"CH_pz.rebo_nkp225",
                     "kolmogorov crespi":"CC_QMC.KC",
                     "KC inspired":"KC_insp.txt",
                     "Pz KC inspired":"KC_insp_pz.txt",
                     "Pz KC inspired nkp225":"KC_insp_pz.txt_nkp225"
                    }
        if type(restart_file)==str:
            f = open(restart_file,'r')
            self.model_dict = json.load(f)
            if self.model_dict["tight binding parameters"] == None:
                self.use_tb=False
            else:
                self.use_tb=True

        else:
            self.create_model(self.model_dict)
         
        self.pylammps_started = False
        self.use_overlap = use_overlap

        self.models_hopping_functions_interlayer = {'letb':letb_interlayer,
                                    'mk':mk,
                                    'popov':popov_hopping,
                                    "nn":nn_hop}
        self.models_overlap_functions_interlayer = {'popov':popov_overlap}
        self.models_cutoff_interlayer={'letb':10,
                                'mk':10,
                                'popov':5.29,
                                "nn":3}
        self.models_self_energy = {'letb':0,
                            'mk':0,
                            'popov':-5.2887,
                            "nn":0}
        self.models_hopping_functions_intralayer = {'letb':letb_intralayer,
                                        'mk':mk,
                                        'porezag':porezag_hopping,
                                        "nn":nn_hop}
        self.models_overlap_functions_intralayer = {'porezag':porezag_overlap}
        self.models_cutoff_intralayer={'letb':10,
                                'mk':10,
                                'porezag':3.7,
                                "nn":4.4}

         
    def init_pylammps(self,atoms):
        """ create pylammps object and calculate corrective potential energy 
        """
        ntypes = len(set(atoms.get_chemical_symbols()))
        data_file = os.path.join(self.output,"tegt.data")
        ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
        L = PyLammps(verbose=False)
        L.command("units		metal")
        L.command("atom_style	full")
        L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
        L.command("box tilt large")
    
        L.command("read_data "+data_file)

        L.command("group top type 1")
        L.command("mass 1 12.0100")
            
        if ntypes ==2:
           L.command("group bottom type 2")
           L.command("mass 2 12.0100")
    
        L.command("velocity	all create 0.0 87287 loop geom")
        # Interaction potential for carbon atoms
        ######################## Potential defition ########################
        
        if ntypes ==2 and self.use_tb:
            L.command("pair_style       hybrid/overlay reg/dep/poly 10.0 0 airebo 3")
            L.command("pair_coeff       * *   reg/dep/poly  "+self.kc_file+"   C C") # long-range 
            #L.command("pair_coeff      * * airebo/sigma "+self.rebo_file+" C C")
            L.command("pair_coeff      * * airebo "+self.rebo_file+" C C")
        elif ntypes==2 and not self.use_tb:
            L.command("pair_style       hybrid/overlay kolmogorov/crespi/full 10.0 0 rebo")
            L.command("pair_coeff       * *   kolmogorov/crespi/full  "+self.kc_file+"   C C") # long-range
            L.command("pair_coeff      * * rebo "+self.rebo_file+" C C")
        elif ntypes==1 and self.use_tb:
            L.command("pair_style       airebo 3")
            L.command("pair_coeff      * * "+self.rebo_file+" C")
        else:
            L.command("pair_style       rebo")
            L.command("pair_coeff      * * "+self.rebo_file+" C")

        ####################################################################

        L.command("timestep 0.00025")
        L.command("thermo 1")
        L.command("fix 1 all nve")
        return L
    
    def run_lammps(self,atoms):
        """ evaluate corrective potential energy, forces in lammps 
        """
        if not atoms.has("mol-id"):
            mol_id = np.ones(len(atoms),dtype=np.int8)
            sym = atoms.get_chemical_symbols()
            top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
            mol_id[top_layer_ind] += 1
            atoms.set_array("mol-id",mol_id)
        #update atom positions in lammps object, need to make sure pylammps object is only initialized on rank 0 so I don't have to keep writing data files
        #if not self.pylammps_started:
        self.L = self.init_pylammps(atoms)
        #pos = atoms.positions
        #for i in range(atoms.get_global_number_of_atoms()):
        #    self.L.atoms[i].position = pos[i,:]
            
        forces = np.zeros((atoms.get_global_number_of_atoms(),3))
        
        self.L.run(0)
        pe = self.L.eval("pe")
        ke = self.L.eval("ke")
        for i in range(atoms.get_global_number_of_atoms()):
            forces[i,:] = self.L.atoms[i].force
        del self.L
        return forces,pe,pe+ke
    
    def get_tb_forces(self,kpoints):
        def func(i):
            #get energy and force at a single kpoint from gpu
            kpoint_slice = kpoints[i,:]
            kpoint_slice = np.asarray(kpoint_slice)
            recip_cell = self.get_recip_cell(self.cell)
            
            if kpoint_slice.shape == (3,):
                kpoint_slice = kpoint_slice.reshape((1, 3))
            
            kpoint_slice = kpoint_slice @ recip_cell
            nkp = kpoint_slice.shape[0]
            natoms = self.positions.shape[0]

            Energy = 0
            Forces = np.zeros((natoms, 3), dtype=np.complex64)
            for k in range(nkp):
                Ham,Overlap = self.gen_ham_ovrlp(kpoint_slice[k,:])
                if self.use_overlap:
                    eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
                else:
                    eigvalues,eigvectors = np.linalg.eigh(Ham)
                nocc = int(natoms / 2)
                Energy += 2 * np.sum(eigvalues[:nocc])

                Forces += self.get_hellman_feynman(eigvalues,eigvectors,kpoint_slice[k,:] )
                return Energy,Forces
        return func

    
    def get_tb_bands(self,positions,atom_types,cell,kpoints,tbparams):
        """get band structure for a given system and path in kspace.
         
        :param atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

        :param mol_id: (np.ndarray [Natoms,]) atom types expressed as integers

        :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

        :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

        :param params_str: (str) specify which tight binding model to use
        
        :returns: (np.ndarray [Number of eigenvalues, number of kpoints])"""
        def func(i):
            kpoint_slice = kpoints[i,:]
            atom_positions = np.asarray(positions)
            cell = np.asarray(cell)
            kpoint_slice = np.asarray(kpoint_slice)
            mol_id = np.asarray(atom_types)

            recip_cell = self.get_recip_cell(cell)
            if kpoint_slice.ndim == 1:
                kpoint_slice = np.reshape(kpoint_slice, (1,3))
            kpoint_slice = kpoint_slice @ recip_cell.T
            natoms = atom_positions.shape[0]
            nkp = kpoint_slice.shape[0]
            evals = np.zeros((natoms, nkp))
            evecs = np.zeros((natoms, natoms, nkp), dtype=np.complex64)
            
            for k in range(nkp):
                Ham,Overlap = self.gen_ham_ovrlp(kpoint_slice[k,:])
                if self.use_overlap:
                    eigvalues, eigvectors = spla.eigh(Ham,b=Overlap)
                else:
                    eigvalues, eigvectors = np.linalg.eigh(Ham)
                evals[:, k] = eigvalues
                evecs[:, :, k] = eigvectors
            return evals,evecs
        return func
    
    def reduce_bands(self,results,return_evecs=False):
        evals = np.zeros((self.natoms,self.nkp))
        if return_evecs:
            evecs = np.zeros((self.natoms,self.natoms,self.nkp),dtype=complex)
        i=0
        for eva,eve in results:
            evals[:,i] = np.squeeze(eva)
            if return_evecs:
                evecs[:,:,i] = np.squeeze(eve)
            i+=1
        if return_evecs:
            return evals, evecs
        else:
            return evals

    def reduce_energy(self,results):
        total_force = np.zeros((self.natoms,3))
        total_energy = 0
        
        for e, f in results:
            total_force += f.real
            total_energy += e
        return total_energy, total_force
    
    def run_tight_binding(self,atoms,force_type="force"):
        """get total tight binding energy and forces, using either hellman-feynman theorem or finite difference (expensive). 
        This function handles parallelization over kpoints.

        :param atoms: (ase.atoms object) must have array "mol-id" specifying the tight binding types
        
        :returns: tuple(float, np.ndarray [number of atoms, 3]) tight binding energy, tight binding forces"""
        self.positions = atoms.positions
        self.atom_types = atoms.get_array("mol-id")
        self.cell = atoms.get_cell()
        if force_type == "force":
            tb_fxn = self.get_tb_forces(self.kpoints)
        tb_energy = 0
        tb_forces = np.zeros((atoms.get_global_number_of_atoms(),3))
        self.natoms = len(atoms)
        print("nkp = ",self.nkp)
        #this works across multiple nodes

        #dask
        if self.parallel=="dask":
            cluster = LocalCluster()
            client = dask.distributed.Client(cluster)
            start_time = time()

            futures = client.map(tb_fxn, np.arange(self.nkp))
            tb_energy, tb_forces  = client.submit(self.reduce_energy, futures).result()
            end_time = time()

            client.shutdown()
            print("time for tight binding = ",end_time - start_time)

        #serial
        elif self.parallel=="serial":
            results = []
            for i in range(self.nkp):
                e,f = tb_fxn(i)
                results.append((e,f))
            tb_energy, tb_forces = self.reduce_energy(results)
        #joblib
        elif self.parallel=="joblib":
            #ncpu = joblib.cpu_count()
            output = Parallel(n_jobs=self.nkp)(delayed(tb_fxn)(i) for i in range(self.nkp))
            for i in range(self.nkp):
                tb_energy += np.squeeze(output[i][0])
                tb_forces += np.squeeze(output[i][1].real)
        return tb_energy.real/self.nkp, tb_forces.real/self.nkp
    
    def get_band_structure(self,atoms,kpoints):
        """get band structure for a given ase.atoms object and path in kspace.
         
        :param atoms: (ase.atoms object) must have array "mol-id" specifying the tight binding types
        
        :param kpoints: (np.ndarray [Number of kpoints,3]) path in kspace to calculate bands over
        
        :returns: (np.ndarray [Number of eigenvalues, number of kpoints])"""
        self.nkp = np.shape(kpoints)[0]
        sym = atoms.get_chemical_symbols()
        mol_id = atoms.get_array("mol-id")
        tb_fxn = self.get_tb_bands(atoms.positions,mol_id,np.array(atoms.cell),
                                 kpoints,self.model_dict["tight binding parameters"])
        self.natoms = len(atoms)
        evals = np.zeros((len(atoms),self.nkp))
        #evecs = np.zeros((len(atoms),len(atoms),self.nkp),dtype=np.complex64)
        if self.parallel=="dask":
            cluster = dask.distributed.LocalCluster()
            client = dask.distributed.Client(cluster)

            futures = client.map(tb_fxn, np.arange(self.nkp))
            evals  = client.submit(self.reduce_bands, futures).result()
            client.shutdown()
        #serial
        elif self.parallel=="serial":
            results = []
            start_time = time()
            for i in range(self.nkp):
                tmpeval = tb_fxn(i)
                results.append((tmpeval))
            evals = self.reduce_bands(results)
            end_time = time()
            print("time for tight binding = ",end_time - start_time)
        #joblib
        elif self.parallel=="joblib":
            ncpu = joblib.cpu_count()
            output = Parallel(n_jobs=self.nkp)(delayed(tb_fxn)(i) for i in range(self.nkp))
            for i in range(self.nkp):
                evals[:,i] = np.squeeze(output[i][0])
                #evecs[:,:,i] = np.squeeze(output[i][1])
        return evals

    def calculate(self, atoms, properties=None,system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self.use_tb:
            print("getting forces and energies")
            tb_Energy,tb_forces = self.run_tight_binding(atoms)
            Lammps_forces,Lammps_potential_energy,Lammps_tot_energy= self.run_lammps(atoms)
            #Lammps_forces,Lammps_potential_energy,Lammps_tot_energy=  np.zeros((len(atoms),3)), 0 , 0
            self.results['forces'] = tb_forces + Lammps_forces
            self.results['potential_energy'] = tb_Energy + Lammps_potential_energy
            self.results['energy'] = tb_Energy + Lammps_tot_energy
            print("Potential Energy = ",(tb_Energy + Lammps_potential_energy)/len(atoms)," (eV/atom)")

            
        else:
            data_file = os.path.join(self.output,"tegt.data")
            ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
            self.Lammps_forces,self.Lammps_potential_energy,self.Lammps_tot_energy= self.run_lammps(atoms)
            self.results['forces'] = self.Lammps_forces
            self.results['potential_energy'] = self.Lammps_potential_energy
            self.results['energy'] = self.Lammps_tot_energy
            print("Potential Energy = ",( self.Lammps_potential_energy)/len(atoms)," (eV/atom)")

        ase.io.write(os.path.join(self.model_dict["output"],"restart.traj"),atoms)

    def run(self,atoms):
        self.calculate(atoms)
    ##########################################################################
    
    #creating total energy tight binding model, performing calculations w/ model
    
    ##########################################################################
    def create_model(self,input_dict):
        """setup total energy model based on input dictionary parameters
        using mpi and latte will result in ase optimizer to be used,
        else lammps runs relaxation """

        model_dict={"tight binding parameters":{"interlayer":{"hopping":{"model":None,"parameters":None},
                                                              "overlap":{"model":None,"parameters":None}},
                                                "intralayer":{"hopping":{"model":None,"parameters":None},
                                                              "overlap":{"model":None,"parameters":None}}},
             "basis":None,
             "intralayer potential":None,
             "interlayer potential":None,
             "kmesh":(1,1,1),
             "parallel":"joblib",
             "output":".",
             }
        orbs_basis = {"s,px,py,pz":4,"pz":1}
        for k in input_dict.keys():
            model_dict[k] = input_dict[k]

        self.model_dict = model_dict
        self.kpoints = self.k_uniform_mesh(self.model_dict['kmesh'])
        self.nkp = np.shape(self.kpoints)[0]
        self.norbs_per_atoms = orbs_basis[self.model_dict["basis"]]
        self.parallel = model_dict["parallel"]
        if not self.model_dict["tight binding parameters"]:
            use_tb=False
        else:
            use_tb=True
        self.use_tb = use_tb

        if type(self.model_dict["intralayer potential"])==np.ndarray:
            self.rebo_file = "CH_pz.rebo"
            self.write_rebo(self.model_dict["intralayer potential"],self.rebo_file)
            self.rebo_file+="_nkp"+str(self.nkp)

        elif self.model_dict["intralayer potential"] not in self.option_to_file.keys():
            #can give file path to potential file in dictionary
            if os.path.exists(self.model_dict["intralayer potential"]):
                self.rebo_file = self.model_dict["intralayer potential"]
            else:
                print("rebo potential file does not exist")
                exit()
        else:
            self.rebo_file = os.path.join(self.param_root,self.option_to_file[self.model_dict["intralayer potential"]])
            if np.prod(self.model_dict['kmesh'])>1:
                if self.model_dict["intralayer potential"]:
                    if self.model_dict["intralayer potential"].split(" ")[-1]!='nkp'+str(self.nkp):
                        self.model_dict["intralayer potential"] = self.model_dict["intralayer potential"]+' nkp'+str(self.nkp)
                        self.rebo_file+="_nkp"+str(self.nkp)

        if type(self.model_dict["interlayer potential"])==np.ndarray:
            self.kc_file = "KC_insp_pz.txt"
            self.write_kcinsp(self.model_dict["interlayer potential"],self.kc_file)
            self.kc_file+="_nkp"+str(self.nkp)
        
        elif self.model_dict["interlayer potential"] not in self.option_to_file.keys():
            #can give file path to potential file in dictionary
            if os.path.exists(self.model_dict["interlayer potential"]):
                self.kc_file = self.model_dict["interlayer potential"]
            else:
                print("interlayer potential file does not exist")
                exit()
        else:
            self.kc_file = os.path.join(self.param_root,self.option_to_file[self.model_dict["interlayer potential"]])
            if np.prod(self.model_dict["kmesh"])>1:
                if self.model_dict["interlayer potential"]:
                    if self.model_dict["interlayer potential"].split(" ")[-1]!='nkp'+str(self.nkp):
                        self.model_dict["interlayer potential"] = self.model_dict["interlayer potential"]+' nkp'+str(self.nkp)
                        self.kc_file+="_nkp"+str(self.nkp)
        if not self.use_tb:
            #if tight binding model is not called for override choices and use only classical potentials
            self.kc_file = os.path.join(self.param_root,self.option_to_file["kolmogorov crespi"])
            self.rebo_file = os.path.join(self.param_root,self.option_to_file["Rebo"])

        self.output = self.model_dict["output"]
        if self.output!=".":
            if not os.path.exists(self.output):
                os.mkdir(self.output)
            #call parameter files from a specified directory, necessary for fitting
            subprocess.call("cp "+self.rebo_file+" "+self.output,shell=True)
            subprocess.call("cp "+self.kc_file+" "+self.output,shell=True)
            self.rebo_file = os.path.join(self.output,self.rebo_file.split("/")[-1])
            self.kc_file = os.path.join(self.output,self.kc_file.split("/")[-1])

    ################################################################################################################################
            
    # TB Utils
            
    ################################################################################################################################
            
    
    def gen_ham_ovrlp(self, kpoint):
        """
        builds a hamiltonian and overlap matrix using distance dependent tight binding parameters

        :params atom_positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

        :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

        :params cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

        :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

        :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

        :returns: tuple(np.ndarray [Norbs,Norbs], np.ndarray [Norbs,Norbs]) Hamiltonian, Overlap        
        """
        
        conversion = 1.0/.529177 #[bohr/angstrom] ASE is always in angstrom, while our package wants bohr
        model_type = self.model_dict["tight binding parameters"]
        lattice_vectors = np.asarray(self.cell)*conversion
        atomic_basis = np.asarray(self.positions)*conversion
        kpoint = np.asarray(kpoint)/conversion

        layer_types = np.asarray(self.atom_types)
        layer_type_set = set(layer_types)

        natom = len(atomic_basis)
        diFull = []
        djFull = []
        extended_coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
                diFull += [dx] * natom
                djFull += [dy] * natom
        distances = cdist(atomic_basis, extended_coords)
        
        Ham = self.models_self_energy[model_type["interlayer"]["hopping"]["model"]]*np.eye(natom,dtype=np.complex64)
        if self.use_overlap:
            Overlap = np.eye(natom,dtype=np.complex64)
        else:
            Overlap = np.empty((natom,natom))
        for i_int,i_type in enumerate(layer_type_set):
            for j_int,j_type in enumerate(layer_type_set):
                if i_type==j_type:
                    hopping_model = self.models_hopping_functions_intralayer[model_type["intralayer"]["hopping"]["model"]]
                    cutoff = self.models_cutoff_intralayer[model_type["intralayer"]["hopping"]["model"]] * conversion
                    hopping_params = model_type["intralayer"]["hopping"]["params"]
                    if self.use_overlap:
                        overlap_model = self.models_overlap_functions_intralayer[model_type["intralayer"]["overlap"]["model"]]
                        overlap_params = model_type["intralayer"]["overlap"]["params"]
                else:
                    hopping_model = self.models_hopping_functions_interlayer[model_type["interlayer"]["hopping"]["model"]]
                    cutoff = self.models_cutoff_interlayer[model_type["interlayer"]["hopping"]["model"]] * conversion
                    hopping_params = model_type["interlayer"]["hopping"]["params"]
                    if self.use_overlap:
                        overlap_model = self.models_overlap_functions_interlayer[model_type["interlayer"]["overlap"]["model"]]
                        overlap_params = model_type["interlayer"]["overlap"]["params"]

                i, j = np.where((distances > 0.1)  & (distances < cutoff))
                di = np.array(diFull)[j]
                dj = np.array(djFull)[j]
                i  = np.array(i)
                j  = np.array(j % natom)
                valid_indices = layer_types[i] == i_type
                valid_indices &= layer_types[j] == j_type
                valid_indices &= i!=j

                disp = self.get_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                            i[valid_indices], j[valid_indices])
                phases = np.exp((1.0j)*np.dot(kpoint,disp.T))

                hoppings = hopping_model(disp,params=hopping_params)/2  # Divide by 2 since we are double counting every pair
                Ham[i[valid_indices],j[valid_indices]] += hoppings * phases
                Ham[j[valid_indices],i[valid_indices]] += np.conj(hoppings*phases)
                if self.use_overlap:
                    overlap_elem = overlap_model(disp,params=overlap_params)/2
                    Overlap[i[valid_indices],j[valid_indices]] +=   overlap_elem  * phases
                    Overlap[j[valid_indices],i[valid_indices]] +=  np.conj(overlap_elem * phases) 

        return Ham, Overlap

    def get_hoppings(self,model=None,displacements=None):
        if displacements:
            hopping_model = self.models_hopping_functions_interlayer[model["hopping"]["model"]]
            hopping_params = model["hopping"]["params"]
            return hopping_model(displacements,parameters=hopping_params)
        gamma = np.array([0,0,0])
        Ham,_ = self.gen_ham_ovrlp(gamma)
        i,j = np.where(Ham>0)
        return Ham[i,j], i,j

    def get_hellman_feynman(self, eigvals,eigvec, kpoint):
        """Calculate Hellman-feynman forces for a given system. Uses finite differences to calculate matrix elements derivatives 
        
        :params atomic_basis: (np.ndarray [Natoms,3]) positions of atoms in angstroms

        :params layer_types: (np.ndarray [Natoms,]) atom types expressed as integers

        :params lattice_vectors: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

        :params eigvals: (np.ndarray [natoms,]) band structure eigenvalues of system

        :params eigvec: (np.ndarray [natoms,natoms]) eigenvectors of system

        :params model_type: (str) specify which tight binding model to use. Options: [popov, mk]

        :params kpoint: (np.ndarray [3,]) kpoint to build hamiltonian and overlap with

        :returns: (np.ndarray [natoms,3]) tight binding forces on each atom"""
        #get hellman_feynman forces at single kpoint. 
        #dE/dR_i =  - Tr_i(rho_e *dS/dR_i + rho * dH/dR_i)
        #construct density matrix
        natoms = len(self.atom_types)
        conversion = 1.0/.529177 # ASE is always in angstrom, while our package wants bohr
        lattice_vectors = np.array(self.cell)*conversion
        model_type = self.model_dict["tight binding parameters"]
        atomic_basis = self.positions*conversion
        nocc = natoms//2
        fd_dist = 2*np.eye(natoms)
        fd_dist[nocc:,nocc:] = 0
        occ_eigvals = 2*np.diag(eigvals)
        occ_eigvals[nocc:,nocc:] = 0
        density_matrix =  eigvec @ fd_dist  @ np.conj(eigvec).T
        energy_density_matrix = eigvec @ occ_eigvals @ np.conj(eigvec).T
        tot_eng = 2 * np.sum(eigvals[:nocc])

        Forces = np.zeros((natoms,3))
        layer_type_set = set(self.atom_types)

        diFull = []
        djFull = []
        extended_coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                extended_coords += list(atomic_basis[:, :] + lattice_vectors[0, np.newaxis] * dx + lattice_vectors[1, np.newaxis] * dy)
                diFull += [dx] * natoms
                djFull += [dy] * natoms
        distances = cdist(atomic_basis, extended_coords)

        gradH = np.zeros((len(diFull),natoms,3))
        for i_int,i_type in enumerate(layer_type_set):
            for j_int,j_type in enumerate(layer_type_set):

                if i_type==j_type:
                    hopping_model = self.models_hopping_functions_intralayer[model_type["intralayer"]["hopping"]["model"]]
                    cutoff = self.models_cutoff_intralayer[model_type["intralayer"]["hopping"]["model"]] * conversion
                    hopping_params = model_type["intralayer"]["hopping"]["params"]
                    if self.use_overlap:
                        overlap_model = self.models_overlap_functions_intralayer[model_type["intralayer"]["overlap"]["model"]]
                        overlap_params = model_type["intralayer"]["overlap"]["params"]
                else:
                    hopping_model = self.models_hopping_functions_interlayer[model_type["interlayer"]["hopping"]["model"]]
                    cutoff = self.models_cutoff_interlayer[model_type["interlayer"]["hopping"]["model"]] * conversion
                    hopping_params = model_type["interlayer"]["hopping"]["params"]
                    if self.use_overlap:
                        overlap_model = self.models_overlap_functions_interlayer[model_type["interlayer"]["overlap"]["model"]]
                        overlap_params = model_type["interlayer"]["overlap"]["params"]

                indi, indj = np.where((distances > 0.1) & (distances < cutoff))
                di = np.array(diFull)[indj]
                dj = np.array(djFull)[indj]
                i  = np.array(indi)
                j  = np.array(indj % natoms)
                valid_indices = self.atom_types[i] == i_type
                valid_indices &= self.atom_types[j] == j_type
                disp = self.get_disp(lattice_vectors, atomic_basis, di[valid_indices], dj[valid_indices],
                                            i[valid_indices], j[valid_indices])
                phases = np.exp((1.0j)*np.dot(kpoint,disp.T))

                #check gradients of hoppings via finite difference
                grad_hop = np.zeros_like(disp)
                grad_overlap = np.zeros_like(disp)

                delta = 1e-5
                for dir_ind in range(3):
                    dr = np.zeros(3)
                    dr[dir_ind] +=  delta
                    hop_up = hopping_model(disp+dr[np.newaxis,:],params=hopping_params)
                    hop_dwn = hopping_model(disp-dr[np.newaxis,:],params=hopping_params)
                    grad_hop[:,dir_ind] = (hop_up - hop_dwn)/2/delta

                    overlap_up = overlap_model(disp+dr[np.newaxis,:],params=overlap_params)
                    overlap_dwn = overlap_model(disp-dr[np.newaxis,:],params=overlap_params)

                    grad_overlap[:,dir_ind] = (overlap_up - overlap_dwn)/2/delta

                rho =  density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis] 
                energy_rho = energy_density_matrix[i[valid_indices],j[valid_indices]][:,np.newaxis]
                gradH = grad_hop * phases[:,np.newaxis] * rho
                gradH += np.conj(gradH)
                if self.use_overlap:
                    Pulay =  grad_overlap * phases[:,np.newaxis] * energy_rho
                    Pulay += np.conj(Pulay)

                for atom_ind in range(natoms):
                    use_ind = np.squeeze(np.where(i[valid_indices]==atom_ind))
                    ave_gradH = gradH[use_ind,:]
                    if self.use_overlap:
                        ave_gradS = Pulay[use_ind,:] 
                    if ave_gradH.ndim!=2:
                        Forces[atom_ind,:] -= -ave_gradH.real 
                        if self.use_overlap:
                            Forces[atom_ind,:] -=   ave_gradS.real
                    else:
                        Forces[atom_ind,:] -= -np.sum(ave_gradH,axis=0).real 
                        if self.use_overlap:
                            Forces[atom_ind,:] -=   np.sum(ave_gradS,axis=0).real
        return Forces * conversion

    #################################################################################################################################
            
    # General Utils
            
    #################################################################################################################################

    def get_disp(self,lattice_vectors, atomic_basis, di, dj, ai, aj):
        """ 
        Converts displacement indices to physical distances
        Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)

        dxy - Distance in Bohr, projected in the x/y plane
        dz  - Distance in Bohr, projected onto the z axis
        """
        displacement_vector = di[:, np.newaxis] * lattice_vectors[0] +\
                            dj[:, np.newaxis] * lattice_vectors[1] +\
                            atomic_basis[aj] - atomic_basis[ai]
        return displacement_vector
    
    def get_recip_cell(self,cell):
        """find reciprocal cell given real space cell
        :param cell: (np.ndarray [3,3]) real space cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector
        
        :returns: (np.ndarray [3,3]) reciprocal cell of system where recip_cell[i, j] is the jth Cartesian coordinate of the ith reciprocal cell vector"""
        a1 = cell[:, 0]
        a2 = cell[:, 1]
        a3 = cell[:, 2]

        volume = np.dot(a1, np.cross(a2, a3))

        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume

        return np.array([b1, b2, b3])
    
    def k_uniform_mesh(self,mesh_size):
        r""" 
        Returns a uniform grid of k-points that can be passed to
        passed to function :func:`pythtb.tb_model.solve_all`.  This
        function is useful for plotting density of states histogram
        and similar.

        Returned uniform grid of k-points always contains the origin.

        :param mesh_size: Number of k-points in the mesh in each
          periodic direction of the model.
          
        :returns:

          * **k_vec** -- Array of k-vectors on the mesh that can be
        """
         
        # get the mesh size and checks for consistency
        use_mesh=np.array(list(map(round,mesh_size)),dtype=int)
        # construct the mesh
        
        # get a mesh
        k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
        # normalize the mesh
        norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
        norm=norm.reshape(use_mesh.tolist()+[3])
        norm=norm.transpose([3,0,1,2])
        k_vec=k_vec/norm
        # final reshape
        k_vec=k_vec.transpose([1,2,3,0]).reshape([use_mesh[0]*use_mesh[1]*use_mesh[2],3])
        return k_vec

    def k_path(self,sym_pts,nk,report=False):
        r"""
    
        Interpolates a path in reciprocal space between specified
        k-points.  In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes"),
        and the results can be used to plot the bands along this path.
        
        The interpolated path that is returned contains as
        equidistant k-points as possible.
    
        :param kpts: Array of k-vectors in reciprocal space between
          which interpolated path should be constructed. These
          k-vectors must be given in reduced coordinates.  As a
          special case, in 1D k-space kpts may be a string:
    
          * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
          * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
          * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)
    
        :param nk: Total number of k-points to be used in making the plot.
        
        :param report: Optional parameter specifying whether printout
          is desired (default is True).

        :returns:

          * **k_vec** -- Array of (nearly) equidistant interpolated
            k-points. The distance between the points is calculated in
            the Cartesian frame, however coordinates themselves are
            given in dimensionless reduced coordinates!  This is done
            so that this array can be directly passed to function
            :func:`pythtb.tb_model.solve_all`.

          * **k_dist** -- Array giving accumulated k-distance to each
            k-point in the path.  Unlike array *k_vec* this one has
            dimensions! (Units are defined here so that for an
            one-dimensional crystal with lattice constant equal to for
            example *10* the length of the Brillouin zone would equal
            *1/10=0.1*.  In other words factors of :math:`2\pi` are
            absorbed into *k*.) This array can be used to plot path in
            the k-space so that the distances between the k-points in
            the plot are exact.

          * **k_node** -- Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates.  This array is
            typically used to plot nodes (typically special points) on
            the path in k-space.
        """
    
        k_list=np.array(sym_pts)
    
        # number of nodes
        n_nodes=k_list.shape[0]
    
        mesh_step = nk//(n_nodes-1)
        mesh = np.linspace(0,1,mesh_step)
        step = (np.arange(0,mesh_step,1)/mesh_step)
    
        kvec = np.zeros((0,3))
        knode = np.zeros(n_nodes)
        for i in range(n_nodes-1):
           n1 = k_list[i,:]
           n2 = k_list[i+1,:]
           diffq = np.outer((n2 - n1),  step).T + n1
    
           dn = np.linalg.norm(n2-n1)
           knode[i+1] = dn + knode[i]
           if i==0:
              kvec = np.vstack((kvec,diffq))
           else:
              kvec = np.vstack((kvec,diffq))
        kvec = np.vstack((kvec,k_list[-1,:]))
    
        dk_ = np.zeros(np.shape(kvec)[0])
        for i in range(1,np.shape(kvec)[0]):
           dk_[i] = np.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]
    
        return (kvec,dk_, knode)
    
def write_kcinsp(self,params,kc_file):
    """write kc inspired potential """
    params = params[:9]
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
    

def check_keywords(self,string):
   """check to see which keywords are in string """
   keywords = ['Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'BIJc_CC3','Beta_CC1', 
             'Beta_CC2','Beta_CC3']
   
   for k in keywords:
       if k in string:
           return True, k
      
   return False,k
   
def write_rebo(self,params,rebo_file):
    """write rebo potential given list of parameters. assumed order is
    Q_CC , alpha_CC, A_CC, BIJc_CC1, BIJc_CC2 ,BIJc_CC3, Beta_CC1, Beta_CC2,Beta_CC3
    
    :param params: (list) list of rebo parameters
    """
    keywords = [ 'Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2','BIJc_CC3', 'Beta_CC1', 
              'Beta_CC2', 'Beta_CC3']
    param_dict=dict(zip(keywords,params))
    with open(rebo_file, 'r') as f:
        lines = f.readlines()
        new_lines=[]
        for i,l in enumerate(lines):
            
            in_line,line_key = self.check_keywords(l)
            
            if in_line:
                nl = str(param_dict[line_key])+" "+line_key+" \n"
                new_lines.append(nl)
            else:
                new_lines.append(l)
    with open(rebo_file, 'w') as f:        
        f.writelines(new_lines)
