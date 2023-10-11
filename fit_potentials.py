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
import mlflow

def get_bilayer_atoms(d,disregistry, a=2.462,c=15, zshift='CM'):
    a_nn = a/np.sqrt(3)
    stack = np.array([[0,0],[3*disregistry*a_nn+a_nn,0]])
    atoms = fg.shift.make_graphene(stacking=stack,cell_type='rect',
                        n_layer=2,n_1=5,n_2=5,lat_con=a,
                        sep=d,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=c)
    return atoms

def get_monolayer_atoms(dx,dy,a=2.462):
    atoms=fg.shift.make_layer("A","rect",4,4,a,7.0,"C",12.01,1)
    curr_cell=atoms.get_cell()
    curr_cell[-1,-1]=14
    atoms.set_cell(curr_cell)
    return ase.Atoms(atoms) 
    
def write_kcinsp(params):
    params = params[:9]
    params = " ".join([str(x) for x in params])
    
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open("KC_insp_pz.txt", 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
    

def format_params(params, sep=' ', prec='.15f'):
    l = [f'{param:{prec}}' for param in params]
    s = sep.join(l)
    return s

def check_keywords(string):
   """check to see which keywords are in string """
   keywords = ['Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'Beta_CC1', 
             'Beta_CC2']
   
   for k in keywords:
       if k in string:
           return True, k
      
   return False,k
   
def write_rebo(params):
    """write rebo potential given list of parameters. assumed order is
    Q_CC , alpha_CC, A_CC, BIJc_CC1, BIJc_CC2 , Beta_CC1, Beta_CC2
    
    :param params: (list) list of rebo parameters
    """
    keywords = [ 'Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2', 'Beta_CC1', 
              'Beta_CC2']
    param_dict=dict(zip(keywords,params))
    with open("CH_pz.rebo", 'r') as f:
        lines = f.readlines()
        new_lines=[]
        for i,l in enumerate(lines):
            
            in_line,line_key = check_keywords(l)
            
            if in_line:
                nl = str(param_dict[line_key])+" "+line_key+" \n"
                new_lines.append(nl)
            else:
                new_lines.append(l)
    with open("CH_pz.rebo", 'w') as f:        
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
            self.potential_file = "CH_pz.rebo"
        elif self.potential=="KC inspired":
            self.write_potential = write_kcinsp
            self.output = "KCinsp"
            self.potential_file = "KC_insp_pz.txt"
        
    def objective(self,params):
        energy = []
        rms=[]
        E0 = params[-1]
        for row in self.db.select():
    
            atoms = self.db.get_atoms(id = row.id)
            self.write_potential(params[:-1])
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
        with open(self.output+"_rms.txt","a+") as f:
            f.write(str(rms)+" "+wp+"\n")
        return rms
    
    def fit(self,p0,min_type="Nelder-Mead"):
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

        self.write_potential(popt.x)
        subprocess.call("cp "+self.potential_file+" "+self.potential_file+"_final_version",shell=True)
        return popt
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--tbmodel',type=str,default='popov')
    parser.add_argument('-t','--type',type=str,default='interlayer')
    parser.add_argument('-g','--gendata',type=str,default="False")
    parser.add_argument('-f','--fit',type=str,default='True')
    parser.add_argument('-s','--test',type=str,default='False')
    parser.add_argument('-o','--output',type=str,default="fit_"+args.tbmodel+"_"+args.type)
    args = parser.parse_args() 

    #gen_data_bilayer = False
    #gen_data_monolayer=False
    #fit_KC_model = False
    #fit_rebo_model = False
    #test_kc_model=True
    #test_rebo_model=False
    
    model_dict = dict({"tight binding parameters":args.tbmodel, 
                          "basis":"pz",
                          "kmesh":(15,15,1),
                          "intralayer potential":"Pz rebo",
                          "interlayer potential":"Pz KC inspired",
                          'label':args.output})
    
    calc_obj = TEGT_calc.TEGT_Calc(model_dict)
    
    if args.gendata==bool(True) and args.type=="interlayer":
    #if gen_data_bilayer:
        db = ase.db.connect('data/bilayer_nkp225.db')
        df = pd.read_csv('data/qmc.csv')
        for i, row in df.iterrows():
            print(i)
            atoms = get_bilayer_atoms(row['d'], row['disregistry'])
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            db.write(atoms,data={"total_energy":row["energy"],'tb_energy':tb_energy/len(atoms)})

    if args.type=="interlayer" and bool(args.fit):      
    #if fit_KC_model:
        db = ase.db.connect('data/bilayer_nkp225.db')
        E0 = -154
        p0= [4.728912880179687, 32.40993806452906, -20.42597835994438,
             17.187123897218854, -23.370339868938927, 3.150121192047732,
             1.6724670937654809 ,13.646628785353208, 0.7907544823937784,E0]
        potential = "KC inspired"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential)
        pfinal = fitting_obj.fit(p0)
        print(pfinal.x)

    if args.gendata==bool(True) and args.type=="intralayer":   
    #if gen_data_monolayer:
        db = ase.db.connect('data/monoayer_nkp225.db')
        file_list = glob.glob("../tBLG_DFT/calc*",recursive=True)
        for f in file_list:
            atoms = ase.io.read(os.path.join(f,"log"),format="espresso")
            total_energy = atoms.get_potential_energy()
            tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            db.write(atoms,data={"total_energy":total_energy/len(atoms),'tb_forces':tb_forces,'tb_energy':tb_energy/len(atoms)})

    if args.type=="intralayer" and bool(args.fit):
    #if fit_rebo_model:
        db = ase.db.connect('data/monolayer_nkp225.db')
        E0 = 0
        p0 = [0.4787439526021916 ,4.763581262711529,10493.065144313845,11193.716433093443,
              -4.082242700692129,4.59957491269822, 0.07885385443664605,E0]
        potential = "rebo"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential,fit_forces=True)
        pfinal = fitting_obj.fit(p0)
        print(pfinal.x)

    if args.type=="interlayer" and bool(args.test):    
    #if test_kc_model:
        stacking_ = ["AB","SP","Mid","AA"]
        disreg_ = [0 , 0.16667, 0.5, 0.66667]
        colors = ["blue","red","black","green"]
        d_ = np.linspace(3,5,5)
        df = pd.read_csv('data/qmc.csv')
        
        d_ab = df.loc[df['disregistry'] == 0, :]
        min_ind = np.argmin(d_ab["energy"].to_numpy())
        E0_qmc = d_ab["energy"].to_numpy()[min_ind]
        d = d_ab["d"].to_numpy()[min_ind]
        disreg = d_ab["disregistry"].to_numpy()[min_ind]

        E0_tegt = -144.17107834841926
        
        for i,stacking in enumerate(stacking_):
            energy_dis_tegt = []
            energy_dis_qmc = []
            d_ = []
            dis = disreg_[i]
            d_stack = df.loc[df['stacking'] == stacking, :]
            for j, row in d_stack.iterrows():
                atoms = get_bilayer_atoms(row["d"],dis)
                atoms.calc = calc_obj
                total_energy = (atoms.get_potential_energy())/len(atoms) - E0_tegt - E0_qmc
                qmc_total_energy = (row["energy"]-E0_qmc)

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                d_.append(row["d"])

            plt.plot(d_,energy_dis_tegt,label=dis+" tegt",c=colors[i])
            plt.scatter(d_,energy_dis_qmc,label=dis+" qmc",c=colors[i])
        plt.xlabel("interlayer distance (Angstroms)")
        plt.ylabel("interlayer energy (eV)")
        plt.legend()
        plt.savefig("kc_insp_test.png")
        plt.show()
        
    if args.type=="intralayer" and bool(args.test):
    #if test_rebo_model:
        a = 2.462
        n=10
        lat_con_list = np.linspace((1-0.005)*a,(1.005)*a,n)
        lat_con_energy = np.zeros(n)
        for i,lat_con in enumerate(lat_con_list):
            atoms = get_monolayer_atoms(0,0,a=lat_con)
            atoms.calc = calc_obj
            total_energy = atoms.get_potential_energy()/len(atoms)
            lat_con_energy[i] = total_energy
        plt.plot(lat_con_list,lat_con_energy,label = "rebo fit")
        plt.xlabel("lattice constant (angstroms)")
        plt.ylabel("energy (eV)")
        plt.legend()
        plt.savefig("rebo_test.png")
        plt.show()

