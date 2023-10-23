# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:46:47 2023

@author: danpa
"""

from TEGT import TEGT_calc
from ase import Atoms
from ase.optimize import FIRE
import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import subprocess
import os
import lammps_logfile

def get_atom_pairs(n,a):
    L=n*a+10
    sym=""
    pos=np.zeros((int(2*n),3))
    for i in range(n):
        sym+="BB"
        pos[i,:] = np.array([i*a,0,0])
        pos[i+n,:] = np.array([i*a,a,0])
    #'BBBBTiTiTiTi'(0,a,0),(a,2*a,0),(2*a,3*a,0),(3*a,4*a,0)
    atoms = Atoms(sym,positions=pos, #,(2*a,0,0),(a,a,0)],
                  cell=[L,L,L])
    return atoms

def get_stack_atoms(n,a):
    sep=3.35
    atoms=fg.shift.make_graphene(stacking=["A","B"],cell_type='rect',
                            n_layer=2,n_1=n,n_2=n,lat_con=a,
                            sep=3.35,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=15)
    return atoms

def get_twist_geom(t,sep,a=2.46):
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=20)
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
        
    # plot first and second band
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
    test_relaxation=True
    test_scaling=False
    theta = 21.78
    
    
    model_dict = dict({"tight binding parameters":"popov", 
                          "basis":"pz",
                          "kmesh":(1,1,1),
                          "intralayer potential":"Pz rebo",
                          "interlayer potential":"Pz KC inspired",
                          'output':"theta_21_78"})
    
    calc_obj = TEGT_calc.TEGT_Calc(model_dict)
    
    if test_tbforces:
        #test forces pairwise
        a_ = np.linspace(2.2,2.6,3)
        n_ = np.arange(2,4,1)
        n=3
        #a = 2.46
        for i,a in enumerate(a_):
            for j,n in enumerate(n_):
                #atoms = get_stack_atoms(n,a)
                atoms = get_atom_pairs(n,a)
                print("n= ",n," a= ",a)
                tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
                #print("Julia forces = ",tb_forces)
                #print("Julia forces (natoms "+str(n)+")= ",tb_forces[:4,:])
                
                tb_energy,tb_forces_fd = calc_obj.run_tight_binding(atoms,force_type="force_fd")
                #print("Julia forces fd = ",tb_forces_fd)
                #print("Julia forces  fd (natoms "+str(n)+")= ",tb_forces_fd[:4,:])
                
                print("error = ",np.mean((tb_forces_fd[:,0]-tb_forces[:,0])))
                print("ratio x_00= ",tb_forces_fd[0,0]/tb_forces[0,0])
                print("ratio y_00= ", tb_forces_fd[0,1]/tb_forces[0,1])
            
        #exit()
        #test forces
        atoms = get_twist_geom(theta,3.35)
        #test julia interface
        tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
        print("julia force \n",tb_forces)
        print("JULIA <forces> = ",np.mean(np.linalg.norm(tb_forces,axis=1)))
        
        #finite difference forces
        tb_energy,tb_forces_fd = calc_obj.run_tight_binding(atoms,force_type="force_fd")
        print("fd force \n",tb_forces_fd)
        print("JULIA finite difference Energy, <forces> = ",tb_energy," ",np.mean(np.linalg.norm(tb_forces_fd,axis=1)))
        
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
        layer_sep = np.linspace(3,5,10)
        latte_energies_sep = np.array([-91.602691, -91.213895, -91.017339, -90.915491, -90.860638, -90.830305,-90.813764, -90.805065, -90.800582, -90.798475])
        julia_energies_sep = np.zeros(10)
        for i,s in enumerate(layer_sep):
            atoms = get_twist_geom(theta,s)
            #test julia interface
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            julia_energies_sep[i] = tb_energy
            
            
            plt.plot(layer_sep,latte_energies_sep-latte_energies_sep[-1],color="black",label="latte")
            plt.plot(layer_sep,julia_energies_sep-julia_energies_sep[-1],color="red",label="julia")
            plt.legend()
            plt.savefig("layer_sep_energies.png")
            plt.clf()
        
        print("RMSE latte, julia tight binding energy = ",np.linalg.norm(
            (julia_energies_sep-julia_energies_sep[-1])-(latte_energies_sep-latte_energies_sep[-1])))
        print("difference in energies at d=3.44, = ",(julia_energies_sep[2]-julia_energies_sep[-1])
              -(latte_energies_sep[2]-latte_energies_sep[-1]))
    if test_lammps:
        #test lammps interface
        atoms = get_twist_geom(theta,3.35)
        forces,pe,tote = calc_obj.run_lammps(atoms)
        print("<forces>, potEng, TotEng = ",np.mean(np.linalg.norm(forces,axis=1))," ",pe, " ", tote)
        
    if test_bands:
        # srun -num tasks nkp -gpu's per task 1 python test_relax.py ideally
        #test band structure
        atoms = get_twist_geom(theta,3.35)
        Gamma = [0,   0,   0]
        K = [2/3,1/3,0]
        Kprime = [1/3,2/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=100
        kdat = calc_obj.k_path(sym_pts,nk)
        kpoints = kdat[0]
        evals,evecs = calc_obj.get_band_structure(atoms,kpoints)
        plot_bands(evals,kdat,erange=5.0,title=r'$\theta=21.78^o$',figname="theta_21_78.png")
        
    if test_relaxation:
        #test relaxation
        atoms = get_twist_geom(theta,3.35)
        atoms.calc = calc_obj
        calc_folder = "theta_21_78"
        if not os.path.exists(calc_folder):
            os.mkdir(calc_folder)
        #energy = atoms.get_potential_energy()
        dyn = FIRE(atoms,
                   trajectory=os.path.join(calc_folder,"theta_"+str(theta)+".traj"),
                   logfile=os.path.join(calc_folder,"theta_"+str(theta)+".log"))
        dyn.run(fmax=0.00005)
