# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:46:47 2023

@author: danpa
"""

from ase import Atoms
from ase.optimize import FIRE
import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import subprocess
import os
import lammps_logfile
from ase.lattice.hexagonal import Graphite
import TETB_GRAPHENE_GPU.TEGT_calc

def get_atom_pairs(n,a):
    L=n*a+10
    sym=""
    pos=np.zeros((int(2*n),3))
    mol_id = np.ones(int(2*n))
    for i in range(n):
        sym+="BTi"
        mol_id[i+n]=2
        pos[i,:] = np.array([0,0,0])
        pos[i+n,:] = np.array([0,0,(i+1)*a])
    #'BBBBTiTiTiTi'(0,a,0),(a,2*a,0),(2*a,3*a,0),(3*a,4*a,0)
    atoms = Atoms(sym,positions=pos, #,(2*a,0,0),(a,a,0)],
                  cell=[L,L,L])
    atoms.set_array('mol-id',mol_id)
    return atoms

def get_random_atoms(n,a):
    L=n*a+10
    sym=""
    pos=np.zeros((int(2*n),3))
    mol_id = np.ones(int(2*n))
    for i in range(n):
        sym+="BTi"
        mol_id[i+n]=2
        pos[i,:] = np.array([i*a+np.random.rand(),0,0])
        pos[i+n,:] = np.array([i*a+np.random.rand(),a,0])
    #'BBBBTiTiTiTi'(0,a,0),(a,2*a,0),(2*a,3*a,0),(3*a,4*a,0)
    atoms = Atoms(sym,positions=pos, #,(2*a,0,0),(a,a,0)],
                  cell=[L,L,L])
    atoms.set_array('mol-id',mol_id)
    return atoms

def get_stack_atoms(sep,a):
    n=5
    atoms=fg.shift.make_graphene(stacking=["A","B"],cell_type='rect',
                            n_layer=2,n_1=n,n_2=n,lat_con=a,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)
    return atoms

def get_twist_geom(t,sep,a=2.46):
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=20)
    return atoms

def get_graphite(s,a=2.46):
    atoms = Graphite(symbol = 'B',latticeconstant={'a':a,'c':2*s},
               size=(2,2,2))
    pos = atoms.positions
    sym = atoms.get_chemical_symbols()
    mean_z = np.mean(pos[:,2])
    top_layer_ind = np.squeeze(np.where(pos[:,2]>mean_z))
    mol_id = np.ones(len(atoms),dtype=np.int64)
    mol_id[top_layer_ind] =2
    atoms.set_array('mol-id',mol_id)
    for ind in top_layer_ind:
        sym[ind] = "Ti"
    atoms.set_chemical_symbols(sym)

    return atoms

def plot_bands(all_evals,kdat,efermi=None,erange=1.0,colors=['black'],title='',figname=None):
    (kvec,k_dist, k_node) = kdat
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
    ax.set_title(title)
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")
    
    
    if not efermi:
        nbands = np.shape(all_evals)[0]
        efermi = np.mean([all_evals[nbands//2,0],all_evals[(nbands-1)//2,0]])
        fermi_ind = (nbands)//2
    else:
        ediff = np.array(all_evals).copy()
        ediff -= efermi
        fermi_ind = np.argmin(np.abs(ediff))-1

    for n in range(np.shape(all_evals)[0]):
        ax.plot(k_dist,all_evals[n,:]-efermi,c=colors[0])
        
    # make an PDF figure of a plot
    fig.tight_layout()
    ax.set_ylim(-erange,erange)
    fig.savefig(figname)
    plt.clf()
    
   
if __name__=="__main__":
    test_tbforces=False
    test_tbenergy=False
    test_lammps=False
    test_bands=False
    test_relaxation=False
    test_scaling=False
    test_kpoints=True
    theta = 21.78
    #theta = 5.09
    
    
    model_dict = dict({"tight binding parameters":{"interlayer":"popov","intralayer":"porezag"}, 
                          "basis":"pz",
                          "kmesh":(1,1,1),
                          "parallel":"dask",
                          "intralayer potential":"Pz rebo",
                          "interlayer potential":"Pz KC inspired",
                          'output':"theta_21_78"})
    
    calc_obj = TETB_GRAPHENE_GPU.TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
    csfont = {'fontname':'serif',"size":20} 
    if test_tbforces:
        #test forces pairwise
        a_ = np.linspace(1.2,1.6,3)
        n_ = [1] #np.arange(2,4,1)
        n = 4
        
        for i,a in enumerate(a_):
            #    for j,n in enumerate(n_):
                #atoms = get_stack_atoms(n,a)
            #atoms = get_atom_pairs(n,a)
            atoms = get_random_atoms(n,a)
            pos = atoms.positions
            print("n= ",n," a= ",a)
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            print("hellman-feynman forces = ",np.round(tb_forces,decimals=2))
            #plt.quiver(pos[:,0],pos[:,1],tb_forces[:,0],tb_forces[:,1])
            #plt.savefig("tb_force_quiver_hf"+str(a)+".png")
            #plt.clf()
            #print("Julia forces (natoms "+str(n)+")= ",tb_forces[:4,:])
            
            tb_energy,tb_forces_fd = calc_obj.run_tight_binding(atoms,force_type="force_fd")
            #plt.quiver(pos[:,0],pos[:,1],tb_forces_fd[:,0],tb_forces_fd[:,1])
            #plt.savefig("tb_force_quiver_fd"+str(a)+".png")
            #plt.clf()
            print("finite-diff forces = ",np.round(tb_forces_fd,decimals=2))
            #print("ratio = ",np.nan_to_num(tb_forces_fd/tb_forces))
            print("\n\n\n")
        #print("Julia forces  fd (natoms "+str(n)+")= ",tb_forces_fd[:4,:])
        
        #print("error = ",np.mean((tb_forces_fd[:,0]-tb_forces[:,0])))
        #print("ratio x_00= ",tb_forces_fd[0,0]/tb_forces[0,0])
        #print("ratio y_00= ", tb_forces_fd[0,1]/tb_forces[0,1])
    
        exit()
        #test forces
        atoms = get_twist_geom(theta,3.35)
        #test julia interface
        tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
        print("hellman-feynman force \n",tb_forces)
        print("hellman-feynman <forces> = ",np.mean(np.linalg.norm(tb_forces,axis=1)))
        
        tb_energy,tb_forces_fd = calc_obj.run_tight_binding(atoms,force_type="force_fd")
        print("fd force \n",tb_forces_fd)
        print("finite difference Energy, <forces> = ",tb_energy," ",np.mean(np.linalg.norm(tb_forces_fd,axis=1)))
        
        #print(tb_forces_fd/tb_forces)
        print("force difference/atom between fd and hellman feynman= ",np.mean(np.linalg.norm(tb_forces_fd-tb_forces,axis=1)))
        
       
    if test_scaling:
        import time
        from scipy.optimize import curve_fit
        
        def cubic_function(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        twist_angles = np.array([5.09,2.88,1.47])
        time_vals = np.zeros_like(twist_angles)
        twist_natoms = np.zeros_like(twist_angles)
        for i,t in enumerate(twist_angles):
            print("twist angle ",t)
            tatoms = get_twist_geom(t,3.35)
            #test julia interface
            start_time = time.time()
            tb_energy,tb_forces = calc_obj.run_tight_binding(tatoms)
            end_time = time.time()
            print(end_time - start_time)
            time_vals[i] = end_time - start_time
            twist_natoms[i] = tatoms.get_global_number_of_atoms()
        
        print(time_vals)
        params, covariance = curve_fit(cubic_function, twist_natoms, time_vals)

        # Extract the fitted parameters
        a_fit, b_fit, c_fit, d_fit = params
        
        large_twists_natoms = np.linspace(20,15000,30)
        # Generate the fitted curve using the fitted parameters
        y_fit = cubic_function(large_twists_natoms, a_fit, b_fit, c_fit, d_fit)
        plt.figure(figsize=(8, 6))
        plt.scatter(twist_natoms, time_vals, label='calculated times', color='b', alpha=0.5)
        plt.plot(large_twists_natoms, y_fit, label='Fitted Curve', color='r')
        plt.xlabel('Number of Atoms')
        plt.ylabel('time')
        plt.legend()
        plt.grid(True)
        plt.title('Julia TEGT scaling')
        plt.savefig("julia_scaling.png")
        plt.show()
            
    if test_tbenergy:

        layer_sep = np.array([3.3266666666666667,3.3466666666666667,3.3866666666666667,3.4333333333333336,3.5,3.5733333333333333,3.6466666666666665,3.7666666666666666,3.9466666666666668,4.113333333333333,4.3533333333333335,4.54,4.76,5.013333333333334,5.16])
        #total interlayer energy/atom of graphite from popov paper at 40x40 kpoint grid

        #layer_sep = np.linspace(3,5,10)
        popov_energies_sep = np.array([ 0.0953237410071943, 0.08884892086330941, 0.07877697841726625, 0.06582733812949645, 0.05323741007194249, 0.042086330935251826, 0.03237410071942448, 0.02230215827338132, 0.01151079136690649, 0.007194244604316571, 0.0025179856115108146, 0.0010791366906475058, 0.0007194244604316752, 0.00035971223021584453, 1.3877787807814457e-17])
        julia_energies_sep = np.zeros_like(layer_sep)
        kpoints = calc_obj.k_uniform_mesh(model_dict["kmesh"])
        tb_energy = 0
        for i,s in enumerate(layer_sep):
            #atoms = get_graphite(s)
            #atoms = get_atom_pairs(1,s)
            atoms = get_stack_atoms(s,2.46)
            #atoms = get_twist_geom(21.78,s,a=2.46)
            #test julia interface
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            #exit()
            julia_energies_sep[i] = tb_energy/len(atoms)
            
            
        plt.plot(layer_sep,popov_energies_sep-popov_energies_sep[-1],color="black",label="popov reference")
        plt.plot(layer_sep,(julia_energies_sep-julia_energies_sep[-1]),color="red",label="python") 
        plt.xlabel("interlayer separation (Angstroms)")
        plt.ylabel("interlayer energy (eV)")
        plt.legend()
        plt.savefig("layer_sep_energies.png")
        plt.clf()

        #print("RMSE latte, julia tight binding energy = ",np.linalg.norm(
        #    (julia_energies_sep-julia_energies_sep[-1])-(popov_energies_sep-popov_energies_sep[-1])))
        #print("difference in energies at d=3.44, = ",(julia_energies_sep[2]-julia_energies_sep[-1])
        #      -(popov_energies_sep[2]-popov_energies_sep[-1]))
        
        a_ = np.linspace(2.35,2.6,20)
        s= 3.35
        python_energies = np.zeros_like(a_)
        for i,a in enumerate(a_):
            #atoms = get_graphite(s,a=a)
            atoms = get_stack_atoms(s,a)
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            python_energies[i] = tb_energy/len(atoms)

        plt.plot(a_,python_energies-python_energies[-1])
        plt.savefig("intralayer_energies.png")
        plt.clf()


    if test_kpoints:
        #nkps = np.arange(1,50,1)
        nkps = np.arange(1,16,1)
        nkps = np.append(nkps,50)
        atoms = get_stack_atoms(3.43,2.46)
        tb_energy_k = np.zeros(len(nkps))
        for i,nkp in enumerate(nkps):
            model_dict["kmesh"] = (nkp,nkp,1)    
            calc_obj = TEGT_GPU.TEGT_calc.TEGT_Calc(model_dict)
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            tb_energy_k[i] = tb_energy/len(atoms)
        print(tb_energy_k)
        plt.plot(np.power(nkps,2),np.abs(tb_energy_k-tb_energy_k[-1])*1000,label="TB Energy")
        plt.plot(np.power(nkps,2),3.88*0.01*np.ones_like(nkps),label="Threshold")
        plt.xlim((0,250))
        plt.xlabel("Number of Kpoints",**csfont)
        plt.ylabel("meV/atom",**csfont)
        plt.title("AB Bilayer Graphene",**csfont)
        plt.legend()
        plt.tight_layout()
        plt.savefig("kpoint_convergence.jpg",dpi=1200)
        plt.clf()

    
    
    if test_lammps:
        #test lammps interface
        atoms = get_twist_geom(theta,3.35)
        forces,pe,tote = calc_obj.run_lammps(atoms)
        print("<forces>, potEng, TotEng = ",np.mean(np.linalg.norm(forces,axis=1))," ",pe, " ", tote)
        
    if test_bands:
        # srun -num tasks nkp -gpu's per task 1 python test_relax.py ideally
        #test band structure
        theta=5.09 #21.78
        atoms = get_twist_geom(theta,3.35)
        Gamma = [0,   0,   0]
        K = [2/3,1/3,0]
        Kprime = [1/3,2/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=60
        kdat = calc_obj.k_path(sym_pts,nk)
        kpoints = kdat[0]
        evals,evecs = calc_obj.get_band_structure(atoms,kpoints)
        plot_bands(evals,kdat,erange=5,title=r'$\theta=$'+str(theta)+r'$^o$',figname="theta_"+str(theta)+".png")
        
    if test_relaxation:
        from ase.optimize import BFGS
        #test relaxation
        #theta = 2.88
        theta = 5.09
        atoms = get_twist_geom(theta,3.35)
        calc_folder = "theta_"+str(theta).replace(".","_")
        model_dict = dict({"tight binding parameters":{"interlayer":"popov","intralayer":"porezag"},
                          "basis":"pz",
                          "kmesh":(1,1,1),
                          "parallel":"joblib",
                          "intralayer potential":"Pz rebo",
                          "interlayer potential":"Pz KC inspired",
                          'output':calc_folder})

        calc_obj = TEGT_GPU.TEGT_calc.TEGT_Calc(model_dict)
        #atoms = get_graphite(3.35)
        atoms.calc = calc_obj
        calc_folder = "theta_"+str(theta).replace(".","_")
        #calc_folder = "theta_5_09"
        if not os.path.exists(calc_folder):
            os.mkdir(calc_folder)
        #else:
        #    atoms = ase.io.read(os.path.join(calc_folder,"theta_"+str(theta)+".traj"))
        #energy = atoms.get_potential_energy()
        dyn = FIRE(atoms,
                   trajectory=os.path.join(calc_folder,"theta_"+str(theta)+".traj"),
                   logfile=os.path.join(calc_folder,"theta_"+str(theta)+".log"))
        #dyn = BFGS(atoms,
        #           trajectory=os.path.join(calc_folder,"theta_"+str(theta)+".traj"),
        #           logfile=os.path.join(calc_folder,"theta_"+str(theta)+".log"))
        dyn.run(fmax=0.00005)
