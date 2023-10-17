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
#from julia.api import Julia
#jl = Julia(compiled_modules=False)
#jl.eval('include("TEGT_TB.jl")')
#from julia import Main

import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
import joblib
from joblib import Parallel, delayed
from mpi4py import MPI
import TEGT

#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces in parallel

class TEGT_Calc(Calculator):
    
    implemented_properties = ['energy','forces','potential_energy']
    def __init__(self,model_dict=None,restart_file=None, device_num=1,device_type="cpu",**kwargs):
        Calculator.__init__(self, **kwargs)
        self.model_dict=model_dict
        self.device_num = device_num
        self.device_type = device_type
        self.repo_root = os.path.join("/".join(TEGT.__file__.split("/")[:-1]),"parameters")
        self.option_to_file={
                     "Rebo":"CH.rebo",
                     "Pz pairwise":"pz_pairwise_correction.table",
                     "Pz rebo":"CH_pz.rebo",
                     "Pz rebo nkp225":"CH_pz.rebo_nkp225",
                     "kolmogorov crespi":"fullKC.txt",
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
        
        self.kpoints = self.k_uniform_mesh(self.model_dict['kmesh'])
        self.nkp = np.shape(self.kpoints)[0]
        self.pylammps_started = False
         
    def init_pylammps(self,atoms):
        ntypes = len(set(atoms.get_chemical_symbols()))
        data_file = os.path.join(self.output,"tegt.data")
        ase.io.write(data_file,atoms,format="lammps-data",atom_style = "full")
        L = PyLammps()
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
        
        if ntypes ==2:
            L.command("pair_style       hybrid/overlay reg/dep/poly 10.0 0 rebo")
            L.command("pair_coeff       * *   reg/dep/poly  "+self.kc_file+"   C C") # long-range 
            L.command("pair_coeff      * * rebo "+self.rebo_file+" C C")
        else:
            L.command("pair_style       rebo")
            L.command("pair_coeff      * * "+self.rebo_file+" C")

        ####################################################################

        L.command("timestep 0.00025")
        L.command("thermo 1")
        L.command("fix 1 all nve")
        return L
    
    def run_lammps(self,atoms):
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
        self.L.close()
        return forces,pe,pe+ke
    
    def get_tb_fxn(self,positions,atom_types,cell,kpoints,tbparams,calc_type="force"):
        if calc_type=="force":
            def func(i):
                from julia.api import Julia
                jl = Julia(compiled_modules=False)
                jl.eval('include("TEGT_TB.jl")')
                from julia import Main
                #get energy and force at a single kpoint from julia
                kpoint = kpoints[i,:]
                energy,force = Main.JULIA_get_tb_forces_energy(positions,atom_types,cell,kpoint,tbparams,self.device_num,self.device_type)
                return energy,force
        elif calc_type=="force_fd":
            def func(i):
                from julia.api import Julia
                jl = Julia(compiled_modules=False)
                jl.eval('include("TEGT_TB.jl")')
                from julia import Main
                #get energy and force at a single kpoint from julia
                kpoint = kpoints[i,:]
                energy,force = Main.JULIA_get_tb_forces_energy_fd(positions,atom_types,cell,kpoint,tbparams)
                return energy,force
        elif calc_type=="bands":
            def func(i):
                from julia.api import Julia
                jl = Julia(compiled_modules=False)
                jl.eval('include("TEGT_TB.jl")')
                from julia import Main
                #get energy and force at a single kpoint from julia
                kpoint = kpoints[i,:]
                evals,evecs = Main.JULIA_get_band_structure(positions,atom_types,cell,kpoint,tbparams,self.device_num,self.device_type)
                return evals,evecs
            
        return func
    
    def run_tight_binding(self,atoms,force_type="force"):
        #have julia calculate energies/forces at individual kpoints, let python do parallelization
        tb_fxn = self.get_tb_fxn(atoms.positions,atoms.get_chemical_symbols(),np.array(atoms.cell),self.kpoints,
                                                   self.model_dict["tight binding parameters"],calc_type=force_type)
        tb_energy = 0
        tb_forces = np.zeros((atoms.get_global_number_of_atoms(),3),dtype=complex)
        
        number_of_cpu = joblib.cpu_count()
        kind = np.array(range(self.nkp))
        #use_ind = np.split(kind,number_of_cpu)
        #ndiv = len(use_ind)
        output = Parallel(n_jobs=self.nkp)(delayed(tb_fxn)(i) for i in range(self.nkp))
        for i in range(self.nkp):
            #e,f = tb_fxn(i)
            tb_energy += output[i][0]
            tb_forces += output[i][1]
        return tb_energy.real/self.nkp, tb_forces.real/self.nkp
    
    def get_band_structure(self,atoms,kpoints):
        nkp = np.shape(kpoints)[0]
        tb_fxn = self.get_tb_fxn(atoms.positions,atoms.get_chemical_symbols(),np.array(atoms.cell),
                                 kpoints,self.model_dict["tight binding parameters"],calc_type="bands")
        evals = np.zeros((atoms.get_global_number_of_atoms(),nkp))
        evecs = np.zeros((atoms.get_global_number_of_atoms(),atoms.get_global_number_of_atoms(),nkp),dtype=complex) 
        number_of_cpu = joblib.cpu_count()
        output = Parallel(n_jobs=number_of_cpu)(delayed(tb_fxn)(i) for i in range(nkp))
        for i in range(nkp):
            evals[:,i] = np.squeeze(output[i][0])
            evecs[:,:,i] = np.squeeze(output[i][1])

        return evals,evecs
        
    def calculate(self, atoms, properties=None,
                  system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        #else:
        self.tb_Energy,self.tb_forces = self.run_tight_binding(atoms)
        #if MPI.COMM_WORLD.rank == 0:
        self.Lammps_forces,self.Lammps_potential_energy,self.Lammps_tot_energy= self.run_lammps(atoms)
        #run lammps part first then run latte part. Sum the two
        if self.use_tb:
            self.results['forces'] = self.Lammps_forces + self.tb_forces 
            self.results['potential_energy'] = self.Lammps_potential_energy + self.tb_Energy
            self.results['energy'] = self.Lammps_tot_energy + self.tb_Energy

        else:
            self.results['forces'] = self.Lammps_forces
            self.results['potential_energy'] = self.Lammps_potential_energy
            self.results['energy'] = self.Lammps_tot_energy
                
            #atoms.calc.forces = self.forces
            #atoms.calc.potential_energy = self.potential_energy
        
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
             "output":".",
             } 
        orbs_basis = {"s,px,py,pz":4,"pz":1}
        for k in input_dict.keys():
            model_dict[k] = input_dict[k]
        self.model_dict = model_dict
        self.norbs_per_atoms = orbs_basis[self.model_dict["basis"]]
        if self.model_dict["tight binding parameters"] == None:
            use_tb=False
        else:
            use_tb=True

        self.use_tb = use_tb
        if np.prod(self.model_dict['kmesh'])>1:
            if self.model_dict["intralayer potential"]:
                if self.model_dict["intralayer potential"].split(" ")[-1]!='nkp225':
                    self.model_dict["intralayer potential"] = self.model_dict["intralayer potential"]+' nkp225'
            if self.model_dict["interlayer potential"]:
                if self.model_dict["interlayer potential"].split(" ")[-1]!='nkp225':
                    self.model_dict["interlayer potential"] = self.model_dict["interlayer potential"]+' nkp225'
        
        self.output = self.model_dict["output"]
        if self.output!=".":
            if not os.path.exists(self.output):
                os.mkdir(self.output)
            #call parameter files from a specified directory, necessary for fitting
            subprocess.call("cp "+os.path.join(self.repo_root,self.option_to_file[self.model_dict["interlayer potential"]])+" "+self.output,shell=True)
            subprocess.call("cp "+os.path.join(self.repo_root,self.option_to_file[self.model_dict["intralayer potential"]])+" "+self.output,shell=True)
            self.rebo_file = os.path.join(self.output,self.option_to_file[self.model_dict["intralayer potential"]])
            self.kc_file = os.path.join(self.output,self.option_to_file[self.model_dict["interlayer potential"]])
        else:
            #call parameter files from installed repo directory
            self.rebo_file = os.path.join(self.repo_root,self.option_to_file[self.model_dict["intralayer potential"]])
            self.kc_file = os.path.join(self.repo_root,self.option_to_file[self.model_dict["interlayer potential"]])
                    
        
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
            directly passed to function  :func:`pythtb.tb_model.solve_all`.

        Example usage::
          
          # returns a 10x20x30 mesh of a tight binding model
          # with three periodic directions
          k_vec = my_model.k_uniform_mesh([10,20,30])
          # solve model on the uniform mesh
          my_model.solve_all(k_vec)
        
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
    
        Example usage::
    
          # Construct a path connecting four nodal points in k-space
          # Path will contain 401 k-points, roughly equally spaced
          path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
          (k_vec,k_dist,k_node) = my_model.k_path(path,401)
          # solve for eigenvalues on that path
          evals = tb.solve_all(k_vec)
          # then use evals, k_dist, and k_node to plot bandstructure
          # (see examples)
        
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
