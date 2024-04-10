from ase import Atoms
from ase.optimize import FIRE
import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import os
import TETB_GRAPHENE.TETB_GRAPHENE_calc

def get_twist_geom(t,sep,a=2.46):
    """generate ase.atoms object of twisted bilayer graphene system 
    :param t: (float) twist angle. fg.twist.find_p_q() finds closest commensurate twist angle
    
    :param sep: (float) interlayer separation

    :param a: (float) graphene lattice constant in angstroms
    
    :returns: (ase.atoms object)"""
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

    for n in range(np.shape(all_evals)[0]):
        ax.plot(k_dist,all_evals[n,:]-efermi,c=colors[0])
        
    # make an PDF figure of a plot
    fig.tight_layout()
    ax.set_ylim(-erange,erange)
    fig.savefig(figname)
    plt.clf()

if __name__=="__main__":
    #generate tblg ase.atoms object 
    theta = 5.09
    atoms = get_twist_geom(theta,3.35)

    #setup TETB_GRAPHENE calculator
    calc_folder = "theta_"+str(theta).replace(".","_")
    model_dict = dict({"tight binding parameters":{"interlayer":"popov","intralayer":"porezag"},
                        "basis":"pz",
                        "kmesh":(11,11,1),
                        "parallel":"joblib", #other options for parallel include "dask" (good for multinode cases) and "serial"
                        "intralayer potential":"Pz rebo",
                        "interlayer potential":"Pz KC inspired",
                        'output':calc_folder})
    
    calc_obj = TETB_GRAPHENE.TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
    atoms.calc = calc_obj

    #calculate total energy and forces on system
    total_energy = atoms.get_potential_energy()
    total_forces = atoms.get_forces()

    #calculate just the tight binding energy and forces
    tb_energy, tb_forces = calc_obj.run_tight_binding(atoms)

    #calculate just the residual potential energy from lammps
    Lammps_forces,Lammps_potential_energy,Lammps_tot_energy= calc_obj.run_lammps(atoms)

    #run relaxation
    dyn = FIRE(atoms,
                   trajectory=os.path.join(calc_folder,"theta_"+str(theta)+".traj"),
                   logfile=os.path.join(calc_folder,"theta_"+str(theta)+".log"))
    dyn.run(fmax=0.005)

    #calculate band structure from relaxed structure
    Gamma = [0,   0,   0]
    K = [2/3,1/3,0]
    Kprime = [1/3,2/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=60
    kdat = calc_obj.k_path(sym_pts,nk)
    kpoints = kdat[0]
    evals = calc_obj.get_band_structure(atoms,kpoints)
    plot_bands(evals,kdat,erange=5,title=r'$\theta=$'+str(theta)+r'$^o$',figname="theta_"+str(theta)+".png")
