from kliff.calculators import Calculator as Kliff_calc
from kliff.dataset.weight import MagnitudeInverseWeight
from kliff.loss import Loss
from kliff.models.parameter_transform import LogParameterTransform
from kliff.uq import MCMC, get_T0, autocorr, mser, rhat
from kliff.loss import Loss
from TETB_MODEL_KLIFF import TETB_KLIFF_Model
from schwimmbad import MPIPool
import numpy as np
import subprocess
import time
import datetime
import glob
import h5py
from TETB_LOSS_KLIFF import *
from TETB_slim import *
from BLG_ase_calc import *
import ase.db
from kliff.dataset.extxyz import read_extxyz, write_extxyz
from kliff.dataset.dataset import Configuration
import flatgraphene as fg
from scipy.spatial import distance
from ase.build import make_supercell
import pandas as pd

def get_monolayer_atoms(dx,dy,a=2.462):
    atoms=fg.shift.make_layer("A","rect",4,4,a,7.0,"B",12.01,1)
    curr_cell=atoms.get_cell()
    atoms.set_array('mol-id',np.ones(len(atoms)))
    curr_cell[-1,-1]=14
    atoms.set_cell(curr_cell)
    return ase.Atoms(atoms) 

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

def get_bilayer_atoms(d,disregistry, a=2.46, c=20, sc=5,zshift='CM'):
    '''All units should be in angstroms'''
    symbols = ["C","C","C","C"]
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

def create_Dataset(interlayer_db=None,intralayer_db=None,dset="interlayer"):
    configs = []

    if dset=="interlayer" or dset=="both":
        for i,row in enumerate(interlayer_db.select()):
            atoms = interlayer_db.get_atoms(id = row.id)
            pos = atoms.positions
            mean_z = np.mean(pos[:,2])
            top_ind = np.where(pos[:,2]>mean_z)
            mol_id = np.ones(len(atoms),dtype=np.int64)
            mol_id[top_ind] = 2
            atoms.set_array("mol-id",mol_id)
            sym = ["C" for j in range(len(atoms))]
            atoms.set_chemical_symbols(sym)

            atoms.total_energy = row.data.total_energy
            atoms.tb_energy = row.data.tb_energy

            a_config = Configuration(cell=atoms.get_cell(),
                        species= atoms.get_chemical_symbols(),
                        coords= atoms.positions,
                        PBC= [True,True,True],
                        energy= (row.data.total_energy-row.data.tb_energy)*len(atoms))
            configs.append(a_config)

    if dset=="intralayer" or dset=="both":
        for i,row in enumerate(intralayer_db.select()):
            atoms = intralayer_db.get_atoms(id = row.id)
            atoms.set_array("mol-id",np.ones(len(atoms),dtype=np.int64))
            sym = ["C" for j in range(len(atoms))]
            atoms.set_chemical_symbols(sym)
            atoms.total_energy = row.data.total_energy
            a_config = Configuration(cell=atoms.get_cell(),
                        species= atoms.get_chemical_symbols(),
                        coords= atoms.positions,
                        PBC= [True,True,True],
                        energy= row.data.total_energy*len(atoms))
            configs.append(a_config)

    return configs


if __name__=="__main__":
    """ run mcmc
    $ export MPIEXEC_OPTIONS="--bind-to core --map-by slot:PE=<num_openmp_processes> port-bindings"
    $ mpiexec -np <num_mpi_workers> ${MPIEXEC_OPTIONS} python script.py
    """
    csfont = {'fontname':'serif',"size":18}

    test_intralayer_lat_con=False
    test_intralayer=True
    test_interlayer=True
    convergence_test = False

    model_type = "Classical"

    if model_type=="Classical":
        calc = BLG_classical(parameters=None, output="Test_Classical")

    elif model_type == "TETB":
        rebo_params = np.load("mnml_hopping_rebo_residual_params.npz")["params"]
        kc_params = np.load("mnml_hopping_kcinsp_residual_params.npz")["params"]
        hopping_params = np.load("tb_params.npz")
        interlayer_hopping_params = np.append(hopping_params["Cpp_sigma_interlayer"],hopping_params["Cpp_pi_interlayer"])
        intralayer_hopping_params = np.append(hopping_params["Cpp_sigma_intralayer"],hopping_params["Cpp_pi_intralayer"])
        parameters = np.append(rebo_params,kc_params)
        parameters = np.append(parameters,interlayer_hopping_params)
        parameters = np.append(parameters,intralayer_hopping_params)

        calc = TETB_slim(parameters=parameters, output="Test_TETB") #,tb_model="letb")


    if convergence_test:
        
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
        all_tb_energies = []
        E0_tegt = 0

        ncells = np.arange(1,14,1)
        tb_energy = np.zeros(len(ncells))
        for i,n in enumerate(ncells):
            atoms = get_bilayer_atoms(3.5,0,sc=n)
            tb_energy[i] = calc.get_tb_energy(atoms)/len(atoms)
        plt.plot(ncells,tb_energy)
        plt.savefig("cell_convergence.png")
        plt.clf()


    if test_interlayer:

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
        all_tb_energies = []
        E0_tegt = 1e10
        
        for i,stacking in enumerate(stacking_):
            energy_dis_tegt = []
            energy_dis_qmc = []
            energy_dis_tb = []
            d_ = []
            dis = disreg_[i]
            d_stack = df.loc[df['stacking'] == stacking, :]
            for j, row in d_stack.iterrows():
                #if row["d"] > 5.01:
                #    continue
                atoms = get_bilayer_atoms(row["d"],dis)

                total_energy = (calc.get_total_energy(atoms))/len(atoms)
                
                if total_energy<E0_tegt:
                    E0_tegt = total_energy

                qmc_total_energy = (row["energy"])

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                d_.append(row["d"])

                if model_type == "TETB":
                    tb_energy = calc.get_tb_energy(atoms)/len(atoms)
                    tb_energy /= len(atoms)
                    energy_dis_tb.append(tb_energy)
                    all_tb_energies.append(tb_energy)


            relative_tetb_energies.append(energy_dis_tegt)
            relative_qmc_energies.append(energy_dis_qmc)
            plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " "+model_type,c=colors[i])
            #plt.scatter(np.array(d_),np.array(energy_dis_tb)-(energy_dis_tb[-1]),label=stacking + " TB",c=colors[i],marker=",")
            plt.scatter(np.array(d_),np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])
        
        relative_tetb_energies = np.array(relative_tetb_energies)
        relative_tetb_energies -= np.min(relative_tetb_energies)

        relative_tetb_energies = relative_tetb_energies[relative_tetb_energies>1e-10]

        relative_qmc_energies = np.array(relative_qmc_energies)
        relative_qmc_energies -= np.min(relative_qmc_energies)
        relative_qmc_energies = relative_qmc_energies[relative_qmc_energies>1e-10]

        rms = np.mean(np.abs(relative_tetb_energies-relative_qmc_energies) /relative_qmc_energies)
        

        print("RMS= ",rms)

        plt.xlabel(r"Interlayer Distance ($\AA$)",**csfont)
        plt.ylabel("Interlayer Energy (eV)",**csfont)
        plt.title(model_type,**csfont)
        plt.tight_layout()
        plt.legend()
        plt.savefig("kc_insp_test_"+model_type+".jpg")
        plt.clf()

    if test_intralayer_lat_con:
        a = 2.462
        lat_con_list = np.sqrt(3) * np.array([1.197813121272366,1.212127236580517,1.2288270377733599,1.2479125248508947,\
                            1.274155069582505,1.3027833001988072,1.3433399602385685,1.4053677932405566,\
                            1.4745526838966203,1.5294234592445326,1.5795228628230618])

        lat_con_energy = np.zeros_like(lat_con_list)
        tb_energy = np.zeros_like(lat_con_list)
        rebo_energy = np.zeros_like(lat_con_list)
        dft_energy = np.array([-5.62588911,-6.226154186,-6.804241219,-7.337927988,-7.938413961,\
                            -8.472277446,-8.961917385,-9.251954937,-9.119902805,-8.832030042,-8.432957809])

        for i,lat_con in enumerate(lat_con_list):
        
            atoms = get_monolayer_atoms(0,0,a=lat_con)
            atoms.set_array('mol-id',np.ones(len(atoms),dtype=np.int64))
            print("a = ",lat_con," natoms = ",len(atoms))
            total_energy = calc.get_total_energy(atoms)/len(atoms)
            #tb_energy_geom,tb_forces = calc_obj.run_tight_binding(atoms)
            #tb_energy[i] = tb_energy_geom/len(atoms)
            #lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
            #rebo_energy[i] = total_energy/len(atoms)
            #total_energy = tote + tb_energy_geom
            lat_con_energy[i] = total_energy
        """fit_min_ind = np.argmin(lat_con_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        rebo_params, covariance = curve_fit(quadratic_function, lat_con_list, lat_con_energy, p0=initial_guess)
        rebo_min = np.min(lat_con_energy*len(atoms))

        dft_min_ind = np.argmin(dft_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        dft_params, covariance = curve_fit(quadratic_function, lat_con_list, dft_energy, p0=initial_guess)
        dft_min = dft_params[-1]

        print("rebo fit minimum energy = ",str(rebo_params[-1]))
        print("rebo fit minimum lattice constant = ",str(lat_con_list[fit_min_ind]))
        print("rebo young's modulus = ",str(rebo_params[0]))
        print("DFT minimum energy = ",str(dft_params[-1]))
        print("DFT minimum lattice constant = ",str(lat_con_list[dft_min_ind]))
        print("DFT young's modulus = ",str(dft_params[0]))"""

        plt.plot(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "rebo fit")
        #plt.plot(lat_con_list/np.sqrt(3),tb_energy-tb_energy[fit_min_ind],label = "tight binding energy")
        #plt.plot(lat_con_list/np.sqrt(3),rebo_energy - rebo_energy[fit_min_ind],label="rebo corrective energy")
        plt.plot(lat_con_list/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
        plt.xlabel(r"nearest neighbor distance ($\AA$)")
        plt.ylabel("energy above ground state (eV/atom)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("rebo_lat_con_"+model_type+".jpg")
        plt.clf()

    if test_intralayer:

        db = ase.db.connect('../data/monolayer_nkp121.db')
        energy = []
        
        nconfig=0
        dft_min = 1e8
        for row in db.select():
            if row.data.total_energy<dft_min:
                dft_min = row.data.total_energy
        tegtb_energy = []
        dft_energy = []   
        nn_dist = []
        atoms_id =[]
        unstrained_atoms = get_monolayer_atoms(0,0,a=2.462)
        unstrained_cell = unstrained_atoms.get_cell()
        
        for row in db.select():
    
            atoms = db.get_atoms(id = row.id)
            atoms_id.append(row.id)

            e = calc.get_total_energy(atoms)/len(atoms)
            tegtb_energy.append(e)
            dft_energy.append(row.data.total_energy)
            nconfig+=1

            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            nn_dist.append(average_distance)
        dft_min = np.min(dft_energy)
        rebo_min_ind = np.argmin(tegtb_energy)
        rebo_min = tegtb_energy[rebo_min_ind]

        rms_tetb  = []
        rms_rebo = []
        for i,e in enumerate(tegtb_energy):
            line = np.linspace(0,1,10)
            ediff_line = line*((dft_energy[i]-dft_min) - (e-rebo_min)) + (e-rebo_min)
            tmp_rms = np.linalg.norm((dft_energy[i]-dft_min) - (e-rebo_min))/(dft_energy[i]-dft_min)

            #if tmp_rms >0.15:
            #    del db[atoms_id[i]]
            #    continue
            print("dft energy (eV/atom) = ",dft_energy[i]-dft_min)
            print("tegtb energy (eV/atom) = ",e-rebo_min)
            print("\n")
            average_distance = nn_dist[i]
            if nn_dist[i] > 1.5 or (dft_energy[i]-dft_min)>0.4:
                continue
            rms_tetb.append(tmp_rms)

            if i==0:
                plt.scatter(average_distance,e-rebo_min,color="red",label=model_type)
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue",label="DFT")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
            else:
                plt.scatter(average_distance,e-rebo_min,color="red")
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
        
        print("rms tetb ",rms_tetb)

        rms_tetb = np.array(rms_tetb)
        rms_rebo = np.array(rms_rebo)
        rms_tetb = rms_tetb[rms_tetb<1e3]
        rms_rebo = rms_rebo[rms_rebo<1e3]
        rms_tetb = np.mean(rms_tetb)
        rms_rebo = np.mean(rms_rebo)
        #rms_tetb = np.mean(np.abs(np.array(tegtb_energy)-rebo_min-(np.array(dft_energy)-dft_min)))
        #rms_rebo = np.mean(np.abs(np.array(rebo_energy)-emprebo_min-(np.array(dft_energy)-dft_min)))
        print("average rms tetb = ",rms_tetb)
        
        print("average difference in tetb energy across all configurations = "+str(rms_tetb)+" (eV/atom)")
        print("average difference in rebo energy across all configurations = "+str(rms_rebo)+" (eV/atom)")
        plt.xlabel(r"average nearest neighbor distance ($\AA$)",**csfont)
        plt.ylabel("energy (eV/atom)",**csfont)
        plt.title(model_type,**csfont)

        plt.legend()
        #plt.colorbar().set_label('RMS', rotation=270,**csfont)
        plt.clim((1e-5,1e-4))
        plt.tight_layout()
        plt.savefig("rebo_test_"+model_type+".jpg")
        plt.clf()
