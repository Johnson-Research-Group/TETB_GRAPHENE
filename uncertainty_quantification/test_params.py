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
from reformat_TETB_GRAPHENE_calc import TETB_GRAPHENE_Calc
import pandas as pd
from ase.build import make_supercell
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
def get_bilayer_atoms(d,disregistry, a=2.46, c=20, sc=5,zshift='CM'):
    '''All units should be in angstroms'''
    symbols = ["B","B","Ti","Ti"]
    atoms = ase.Atoms(
        symbols=symbols,
        positions=get_basis(a, d, c, disregistry, zshift=zshift),
        cell=get_lattice_vectors(a, c),
        pbc=[1, 1, 1],
        tags=[0, 0, 1, 1],
        )
    atoms.set_array("mol-id",np.array([1,1,2,2],dtype=np.int8))  
    atoms = make_supercell(atoms, [[sc, 0, 0], [0, sc, 0], [0, 0, 1]])
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
def get_basis(a, d, c, disregistry, zshift='CM'):

    '''
    `disregistry` is defined such that the distance to disregister from AB to AB again is 1.0,
    which corresponds to 3*bond_length = 3/sqrt(3)*lattice_constant = sqrt(3)*lattice_constant
    so we convert the given `disregistry` to angstrom
    '''
    disregistry_ang = 3**0.5*a*disregistry
    orig_basis = np.array([
        [0, 0, 0],
        [0, a/3**0.5, 0],
        [0, a/3**0.5 + disregistry_ang, d],
        [a/2, a/(2*3**0.5) + disregistry_ang, d]
        ])

    # for open boundary condition in the z-direction
    # move the first layer to the middle of the cell
    if zshift == 'first_layer':
        z = c/2
    # or move the center of mass to the middle of the cell
    elif zshift == 'CM':
        z = c/2 - d/2
    shift_vector = np.array([0, 0, z])
    shifted_basis = orig_basis + shift_vector
    return shifted_basis.tolist()

def get_lattice_vectors(a, c):
    return [
        [a, 0, 0],
        [1/2*a, 1/2*3**0.5*a, 0],
        [0, 0, c]
        ]
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
    #ax.set_ylim(-erange,erange)
    fig.savefig(figname)
    plt.clf()

def gsfe_layer_sep(calc):
    stacking_ = ["AB","SP","Mid","AA"]
    disreg_ = [0 , 0.16667, 0.5, 0.66667]
    colors = ["blue","red","black","green"]
    d_ = np.linspace(3,5,5)
    df = pd.read_csv('../data/qmc.csv') 
    d_ab = df.loc[df['disregistry'] == 0, :]
    min_ind = np.argmin(d_ab["energy"].to_numpy())
    E0_qmc = d_ab["energy"].to_numpy()[min_ind]
    d = d_ab["d"].to_numpy()[min_ind]
    disreg = d_ab["disregistry"].to_numpy()[min_ind]
    relative_tetb_energies = []
    relative_qmc_energies = []
    E0_tegt = 0
        
    for i,stacking in enumerate(stacking_):
        energy_dis_tegt = []
        energy_dis_qmc = []
        energy_dis_tb = []
        d_ = []
        dis = disreg_[i]
        d_stack = df.loc[df['stacking'] == stacking, :]
        for j, row in d_stack.iterrows():
            if row["d"] > 5.01:
                continue
            atoms = get_bilayer_atoms(row["d"],dis)
            atoms.calc = calc_obj
            total_energy = (atoms.get_potential_energy())/len(atoms) 
            
            if total_energy<E0_tegt:
                E0_tegt = total_energy
            qmc_total_energy = (row["energy"])

            energy_dis_tegt.append(total_energy)
            energy_dis_qmc.append(qmc_total_energy)
            #energy_dis_tb.append(tb_energy)
            d_.append(row["d"])
                
        relative_tetb_energies.append(energy_dis_tegt)
        relative_qmc_energies.append(energy_dis_qmc)
        plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " TETB",c=colors[i])
        plt.scatter(d_,np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])
        
    plt.xlabel(r"Interlayer Distance ($\AA$)",**csfont)
    plt.ylabel("Interlayer Energy (eV)",**csfont)
    plt.title("TETB(nkp=121)",**csfont)
    plt.tight_layout()
    plt.legend()
    plt.savefig("kc_insp_test_nkp.jpg")
    plt.show()
 
if __name__=="__main__":
    test_tbforces=False
    test_tbenergy=True
    test_lammps=False
    test_bands=False
    vary_params = False
    theta = 21.78
    #theta = 5.09
     
    popov_hopping_params = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,-0.0024695, 0.0003863, -0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983, -0.0046855, 0.0007303,0.0000225, -0.0000393])
    popov_ovrlp_params = np.array([[-0.0571487, -0.0291832, 0.1558650, -0.1665997,0.0921727, -0.0268106, 0.0002240, 0.0040319,-0.0022450, 0.0005596],
                          [0.3797305, -0.3199876, 0.1897988, -0.0754124,0.0156376, 0.0025976, -0.0039498, 0.0020581,-0.0007114, 0.0001427]])

    porezag_hopping_params = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519, -0.0004838, -0.0000906,
        -0.3793837, 0.3204470, -0.1956799, 0.0883986,-0.0300733, 0.0074465, -0.0008563, -0.0004453, 0.0003842, -0.0001855])
    porezag_ovrlp_params=np.array([[-0.1359608, 0.0226235, 0.1406440, -0.1573794,0.0753818, -0.0108677, -0.0075444, 0.0051533,-0.0013747, 0.0000751],
                [0.3715732, -0.3070867, 0.1707304, -0.0581555,0.0061645, 0.0051460, -0.0032776, 0.0009119,-0.0001265, -0.000227]])
   
    
    
    intralayer_potential = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,40.609282094464604,
        4.62769509546907,0.7945927858501145,2.2242248220983427])
    interlayer_potential = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
        1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
    hopping_params = np.load("tb_params.npz") 
    interlayer_hopping_params = np.append(hopping_params["Cpp_sigma_interlayer"],hopping_params["Cpp_pi_interlayer"])
    print("popov ",popov_hopping_params)
    print("interlayer ",interlayer_hopping_params)
    intralayer_hopping_params = np.append(hopping_params["Cpp_sigma_intralayer"],hopping_params["Cpp_pi_intralayer"])

    model_dict = dict({"tight binding parameters":{"interlayer":{"hopping":
        {"model":"popov","params":np.vstack((interlayer_hopping_params[:10],interlayer_hopping_params[10:])),"rcut":10},
        "overlap":{"model":"popov","params":None}},
        "intralayer":{"hopping":{"model":"porezag","params":np.vstack((intralayer_hopping_params[:10],intralayer_hopping_params[10:])),"rcut":7}},
        "overlap":{"model":"porezag","params":None}},
        "basis":"pz",
        "kmesh":(11,11,1),
        "parallel":"joblib",
        "intralayer potential":intralayer_potential,
        "interlayer potential":interlayer_potential,
        'output':"test_output"})
    
    calc_obj = TETB_GRAPHENE_Calc(model_dict,use_overlap=False)
    csfont = {'fontname':'serif',"size":20} 
    if test_tbforces:
        #test forces pairwise
        a_ = np.linspace(1.2,1.6,3)
        n_ = [1] #np.arange(2,4,1)
        n = 4
        
        for i,a in enumerate(a_):
            #    for j,n in enumerate(n_):
            atoms = get_stack_atoms(n,a)
            pos = atoms.positions
            print("n= ",n," a= ",a)
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            print("hellman-feynman forces = ",np.round(tb_forces,decimals=2))

            print("\n\n\n")

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

    
    
    if test_lammps:
        #test lammps interface
        atoms = get_twist_geom(theta,3.35)
        forces,pe,tote = calc_obj.run_lammps(atoms)
        print("<forces>, potEng, TotEng = ",np.mean(np.linalg.norm(forces,axis=1))," ",pe, " ", tote)
        
    if test_bands:
        # srun -num tasks nkp -gpu's per task 1 python test_relax.py ideally
        #test band structure
        theta=21.78
        atoms = get_twist_geom(theta,3.35)
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

    if vary_params:
        import ase.db
        scale = 1e-2
        nkp = 121
        kmesh = (int(np.sqrt(nkp)),int(np.sqrt(nkp)),1)
        db = ase.db.connect('../data/bilayer_nkp'+str(nkp)+'.db')
        model_dict = dict({"tight binding parameters":{"interlayer":{"hopping":{"model":"popov","params":popov_hopping_params},
                                                              "overlap":{"model":"popov","params":None}},
                                                "intralayer":{"hopping":{"model":"porezag","params":None},
                                                              "overlap":{"model":"porezag","params":None}}},
            "basis":"pz",
            "kmesh":kmesh,
            "parallel":"joblib",
            "intralayer potential":"Pz rebo",
            "interlayer potential":"Pz KC inspired",
            'output':"TETB_gsfe"})
    
        calc_obj = reformat_TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)

        gsfe_layer_sep(calc_obj)

