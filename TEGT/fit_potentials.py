# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:39:20 2023

@author: danpa
"""

import TEGT_calc
import flatgraphene as fg
import numpy as np
import ase.io
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.optimize
import h5py
import argparse
import subprocess
import ase.db
import glob
#import mlflow
from ase.build import make_supercell
from scipy.optimize import curve_fit
from scipy.spatial import distance

def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

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

def get_monolayer_atoms(dx,dy,a=2.462):
    atoms=fg.shift.make_layer("A","rect",4,4,a,7.0,"B",12.01,1)
    curr_cell=atoms.get_cell()
    curr_cell[-1,-1]=14
    atoms.set_cell(curr_cell)
    return ase.Atoms(atoms) 
    
def write_kcinsp(params,kc_file):
    params = params[:9]
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
    

def format_params(params, sep=' ', prec='.15f'):
    l = [f'{param:{prec}}' for param in params]
    s = sep.join(l)
    return s

def check_keywords(string):
   """check to see which keywords are in string """
   keywords = ['Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'BIJc_CC3','Beta_CC1', 
             'Beta_CC2','Beta_CC3']
   
   for k in keywords:
       if k in string:
           return True, k
      
   return False,k
   
def write_rebo(params,rebo_file):
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
            
            in_line,line_key = check_keywords(l)
            
            if in_line:
                nl = str(param_dict[line_key])+" "+line_key+" \n"
                new_lines.append(nl)
            else:
                new_lines.append(l)
    with open(rebo_file, 'w') as f:        
        f.writelines(new_lines)

class fit_potentials_tblg:
    def __init__(self,calc_obj, db, potential,fit_forces=False):
        self.calc = calc_obj
        self.db = db
        self.potential = potential
        self.fit_forces = fit_forces
        if self.potential=="rebo":
            self.write_potential = write_rebo
            self.output = "rebo"
            self.potential_file = self.calc.rebo_file
        elif self.potential=="KC inspired":
            self.write_potential = write_kcinsp
            self.output = "KCinsp"
            self.potential_file = self.calc.kc_file
        
    def objective(self,params):
        energy = []
        rms=[]
        E0 = params[-1]
        for row in self.db.select():
    
            atoms = self.db.get_atoms(id = row.id)
            self.write_potential(params[:-1],self.potential_file)
            atoms.calc = self.calc
            lammps_forces,lammps_pe,tote = self.calc.run_lammps(atoms)
            e = (lammps_pe)/len(atoms) + row.data.tb_energy + E0 #energy per atom
            energy.append(e)
            tmp_rms = (e-(row.data.total_energy))
            if self.fit_forces:
                total_forces = lammps_forces + row.data.tb_forces
                tmp_rms += np.linalg.norm(row.data.forces - total_forces)
            rms.append(tmp_rms) #*sigma[i])
        
        rms = np.linalg.norm(rms)
        wp = [str(p) for p in params]
        wp = " ".join(wp)
        with open(os.path.join(self.calc.output,self.output+"_rms.txt"),"a+") as f:
            f.write(str(rms)+" "+wp+"\n")
        return rms
    
    def fit(self,p0,min_type="basinhopping"):
        '''
        bound all params = [0, np.inf]
        '''

        self.min_type=min_type
        if self.min_type=="Nelder-Mead":
            popt = scipy.optimize.minimize(self.objective,p0, method='Nelder-Mead')
        if self.min_type=="basinhopping":
            popt = scipy.optimize.basinhopping(self.objective,p0,niter=5,
                                               minimizer_kwargs={"method":"Nelder-Mead"},
                                               T=100)
        if self.min_type=="global":
            #fit each parameter individually, multiple times
            niter=10
            self.original_p0=p0.copy()
            for n in  range(niter):
                for i,p in enumerate(p0):
                    self.fit_param = i
                    popt = scipy.optimize.minimize(self.objective,p, method='Nelder-Mead')
                    self.original_p0[i] = popt.x
            params = self.original_p0        
            popt = scipy.optimize.minimize(self.objective,params, method='Nelder-Mead')

        self.write_potential(popt.x,self.potential_file)
        subprocess.call("cp "+self.potential_file+" "+self.potential_file+"_final_version",shell=True)
        return popt
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--tbmodel',type=str,default='popov')
    parser.add_argument('-t','--type',type=str,default='interlayer')
    parser.add_argument('-g','--gendata',type=str,default="False")
    parser.add_argument('-f','--fit',type=str,default='True')
    parser.add_argument('-s','--test',type=str,default='False')
    parser.add_argument('-k','--nkp',type=str,default='225')
    parser.add_argument('-o','--output',type=str,default=None)
    args = parser.parse_args() 
    if args.output==None:
        args.output = "fit_"+args.tbmodel+"_"+args.type+"_nkp"+args.nkp
    
    kd = np.sqrt(int(args.nkp))
    kmesh = (kd,kd,1)
    nkp = str(int(np.prod(kmesh)))

    model_dict = dict({"tight binding parameters":args.tbmodel,
                        "basis":"pz",
                        "kmesh":kmesh,
                        "intralayer potential":"Pz rebo",
                        "interlayer potential":"Pz KC inspired",
                        'output':args.output})
    if args.gendata=="True" and args.type=="interlayer":
   
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)

        print("assembling interlayer database")
        db = ase.db.connect('../data/bilayer_nkp'+nkp+'.db')
        df = pd.read_csv('../data/qmc.csv')
        for i, row in df.iterrows():
            print(i)
            atoms = get_bilayer_atoms(row['d'], row['disregistry'])
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            db.write(atoms,data={"total_energy":row["energy"],'tb_energy':tb_energy/len(atoms)})

    if args.type=="interlayer" and args.fit=="True":
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)
        print("fitting interlayer potential")
        db = ase.db.connect('../data/bilayer_nkp'+nkp+'.db')
        E0 = -154
        p0= [4.728912880179687, 32.40993806452906, -20.42597835994438,
             17.187123897218854, -23.370339868938927, 3.150121192047732,
             1.6724670937654809 ,13.646628785353208, 0.7907544823937784,E0]
        potential = "KC inspired"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential)
        pfinal = fitting_obj.fit(p0)
        print(pfinal.x)

    if args.gendata=="True" and args.type=="intralayer":  
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)
        print("assembling intralayer database")
        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        file_list = glob.glob("../../tBLG_DFT/grapheneCalc*",recursive=True)
        for f in file_list:
            print(os.path.join(f,"log"))
            try:
                atoms = ase.io.read(os.path.join(f,"log"),format="espresso-out")
                total_energy = atoms.get_total_energy()
            except:
                print("DFT failed")
                continue
            atoms.symbols = atoms.get_global_number_of_atoms() * "B"
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            db.write(atoms,data={"total_energy":total_energy/len(atoms),'tb_forces':tb_forces,'tb_energy':tb_energy/len(atoms)})

    if args.type=="intralayer" and args.fit=="True":
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)
        print("fitting intralayer potential")
        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        E0 = 0
        #Q_CC , alpha_CC, A_CC, BIJc_CC1, BIJc_CC2 , BIJc_CC3, Beta_CC1, Beta_CC2, Beta_CC3
        p0 = [0.3134602960833 ,4.7465390606595,10953.544162170,12388.79197798,
              17.56740646509,30.71493208065,4.7204523127, 1.4332132499, 1.3826912506,E0]
        potential = "rebo"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential,fit_forces=False)
        pfinal = fitting_obj.fit(p0)
        print(pfinal.x)

    if args.type=="interlayer" and args.test=="True":   
        model_dict = dict({"tight binding parameters":args.tbmodel,
                          "basis":"pz",
                          "kmesh":kmesh,
                          "intralayer potential":os.path.join(args.output,"CH_pz.rebo_nkp225"),
                          "interlayer potential":os.path.join(args.output,"KC_insp_pz.txt_nkp225_final_version"),
                          'output':args.output})
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)

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

        E0_tegt = 0
        
        for i,stacking in enumerate(stacking_):
            energy_dis_tegt = []
            energy_dis_qmc = []
            d_ = []
            dis = disreg_[i]
            d_stack = df.loc[df['stacking'] == stacking, :]
            for j, row in d_stack.iterrows():
                atoms = get_bilayer_atoms(row["d"],dis)
                atoms.calc = calc_obj
                total_energy = (atoms.get_potential_energy())/len(atoms)
                if total_energy<E0_tegt:
                    E0_tegt = total_energy
                qmc_total_energy = (row["energy"])

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                d_.append(row["d"])

            plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " tegt",c=colors[i])
            plt.scatter(d_,np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])
        plt.xlabel("interlayer distance (Angstroms)")
        plt.ylabel("interlayer energy (eV)")
        plt.title("Corrective Interlayer Potential for BLG, num kpoints = "+str(args.nkp))
        plt.legend()
        plt.savefig("kc_insp_test.png")
        plt.show()
        
    if args.type=="intralayer" and args.test=="True":
        model_dict = dict({"tight binding parameters":args.tbmodel,
                          "basis":"pz",
                          "kmesh":kmesh,
                          "intralayer potential":os.path.join(args.output,"CH_pz.rebo_nkp225_final_version"),
                          "interlayer potential":os.path.join(args.output,"KC_insp_pz.txt_nkp225"),
                          'output':args.output})
        calc_obj = TEGT_calc.TEGT_Calc(model_dict)

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
            print("a = ",lat_con," natoms = ",len(atoms))
            atoms.calc = calc_obj
            #total_energy = atoms.get_potential_energy()/len(atoms)
            tb_energy_geom,tb_forces = calc_obj.run_tight_binding(atoms)
            tb_energy[i] = tb_energy_geom/len(atoms)
            lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
            rebo_energy[i] = tote/len(atoms)
            total_energy = tote + tb_energy_geom
            lat_con_energy[i] = total_energy/len(atoms)
        fit_min_ind = np.argmin(lat_con_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        rebo_params, covariance = curve_fit(quadratic_function, lat_con_list, lat_con_energy, p0=initial_guess)

        dft_min_ind = np.argmin(dft_energy)
        initial_guess = (1.0, 1.0, 1.0)  # Initial parameter guess
        dft_params, covariance = curve_fit(quadratic_function, lat_con_list, dft_energy, p0=initial_guess)

        print("rebo fit minimum energy = ",str(rebo_params[-1]))
        print("rebo fit minimum lattice constant = ",str(lat_con_list[fit_min_ind]))
        print("rebo young's modulus = ",str(rebo_params[0]))
        print("DFT minimum energy = ",str(dft_params[-1]))
        print("DFT minimum lattice constant = ",str(lat_con_list[dft_min_ind]))
        print("DFT young's modulus = ",str(dft_params[0]))

        plt.plot(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "rebo fit")
        #plt.plot(lat_con_list/np.sqrt(3),tb_energy-tb_energy[fit_min_ind],label = "tight binding energy")
        #plt.plot(lat_con_list/np.sqrt(3),rebo_energy - rebo_energy[fit_min_ind],label="rebo corrective energy")
        plt.plot(lat_con_list/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
        plt.xlabel("nearest neighbor distance (angstroms)")
        plt.ylabel("energy above ground state (eV/atom)")
        plt.legend()
        

        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        training_data_energy = []
        training_data_nn_dist_ave = []
        for row in db.select():

            atoms = db.get_atoms(id = row.id)
            training_data_energy.append(row.data.total_energy/len(atoms))

            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            training_data_nn_dist_ave.append(average_distance)

        plt.scatter(training_data_nn_dist_ave,training_data_energy-np.min(training_data_energy),label="DFT training data")
        #plt.ylim(0,5)
        plt.savefig("rebo_test.png")
        plt.show()



