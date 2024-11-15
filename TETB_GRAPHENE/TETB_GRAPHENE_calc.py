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
from TETB_GRAPHENE.TETB_GRAPHENE_TB import *
import dask
from dask.distributed import Client, LocalCluster
#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces in parallel

class TETB_GRAPHENE_Calc(Calculator):
    
    implemented_properties = ['energy','forces','potential_energy']
    def __init__(self,model_dict=None,restart_file=None, **kwargs):
        """
        ase calculator object that calculates total energies and forces given a 
        total energy tight binding model. can kmesh to be either (1,1,1), (11,11,1) or (15,15,1), 
        or tight binding model.
        Parameters:
        :param model_dict: (dict) dictionary containing info on tb model, number of kpoints, interatomic corrective potentials
                            defaults: {"tight binding parameters":None, (str) tight binding model
                                        "intralayer potential":None, (str) can be path to REBO potential file or keyword
                                        "interlayer potential":None, (str) can be path to interlayer potential file or keyword
                                        "kmesh":(1,1,1), (tuple) mesh of kpoints
                                        "parallel":serial, (str) type of parallelization to use in tb calculation
                                        "output":".", (str) optional output directory
                                        } 
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
    
    def get_tb_fxn(self,positions,atom_types,cell,kpoints,tbparams,calc_type="force"):
        """select the tight binding calculation to perform.
        :param positions: (np.ndarray [Natoms,3]) positions of atoms in angstroms

        :param atom_types: (np.ndarray [Natoms,]) atom types expressed as integers

        :param cell: (np.ndarray [3,3]) cell of system where cell[i, j] is the jth Cartesian coordinate of the ith cell vector

        :param kpoints: (np.ndarray [number of kpoints,3]) array of kpoints to calculate tight binding calculation over

        :param tbparams: (str) specify which tight binding model to use

        :param calc_type: (str) type of tb calculation. options: force, force_fd (finite difference forces [slow]), bands"""
        if calc_type=="force":
            def func(i):
                #get energy and force at a single kpoint from gpu
                kpoint = kpoints[i,:]
                energy,force = get_tb_forces_energy(positions,atom_types,cell,kpoint,tbparams)
                return energy,force
        elif calc_type=="force_fd":
            def func(i):
                #get energy and force at a single kpoint from gpu
                kpoint = kpoints[i,:]
                energy,force = get_tb_forces_energy_fd(positions,atom_types,cell,kpoint,tbparams)
                return energy,force
        elif calc_type=="bands":
            def func(i):
                #get energy and force at a single kpoint from gpu
                kpoint = kpoints[i,:]
                evals, evecs = calc_band_structure(positions, atom_types, cell, kpoint, tbparams)
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
        sym = atoms.get_chemical_symbols()
        mol_id = atoms.get_array("mol-id")
        tb_fxn = self.get_tb_fxn(atoms.positions,mol_id,np.array(atoms.cell),self.kpoints,self.model_dict["tight binding parameters"],calc_type=force_type)
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
            """
            from dask.distributed import performance_report

            ## some dask computation
            scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")

            client = Client(scheduler_file=scheduler_file)
            from dask import config
            with dask.config.set({'distributed.scheduler.worker-ttl':"1000 minutes"}):
                futures = client.map(tb_fxn, np.arange(self.nkp))
                tb_energy, tb_forces  = client.submit(self.reduce_energy, futures).result()"""

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
        tb_fxn = self.get_tb_fxn(atoms.positions,mol_id,np.array(atoms.cell),
                                 kpoints,self.model_dict["tight binding parameters"],calc_type="bands")
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

        model_dict={"tight binding parameters":None,
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
        if self.model_dict["intralayer potential"] not in self.option_to_file.keys():
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

        if self.model_dict["interlayer potential"] not in self.option_to_file.keys():
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
