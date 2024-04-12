# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:39:20 2023

@author: danpa
"""

from TETB_GRAPHENE import TETB_GRAPHENE_calc
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

def quadratic_function(x, a, c, dx):
    return a * (x-dx)**2 + c

def morse(x, D, a, re, E0):
    return D * np.power((1 - np.exp(-a * (x - re))),2) + E0

def get_binding_energy_sep(d,energy,min_type="quadratic"):
    min_ind = np.argmin(energy)
    D0 = energy[min_ind]
    re0 = d[min_ind]
    initial_guess = [D0, 1.0, re0, energy[-1]]
    params, covariance = curve_fit(morse, d, energy, p0=initial_guess)
    D_fit, a_fit, re_fit, E0_fit = params

    return D_fit, re_fit

def get_energy_min(d,energy):
    min_ind = np.argmin(energy)
    E0 = energy[min_ind]
    dx0 = d[min_ind]
    initial_guess = [1, E0, dx0]
    params, covariance = curve_fit(quadratic_function, d[min_ind-1:min_ind+2], energy[min_ind-1:min_ind+2], p0=initial_guess)
    a_fit, E0_fit, dx_fit = params

    return E0_fit

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
    atoms.set_array('mol-id',np.ones(len(atoms)))
    curr_cell[-1,-1]=14
    atoms.set_cell(curr_cell)
    return ase.Atoms(atoms) 
    
def write_kcinsp(params,kc_file):
    """write kc inspired potential """
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
    def __init__(self,calc_obj, db, potential,optimizer_type="Nelder-Mead",fit_forces=False):
        self.calc = calc_obj
        self.db = db
        self.potential = potential
        self.optimizer_type=optimizer_type
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
        fit_energy = []
        #E0 = params[-1]
        self.write_potential(params,self.potential_file)
        for row in self.db.select():
    
            atoms = self.db.get_atoms(id = row.id)
            atoms.calc = self.calc
            lammps_forces,lammps_pe,tote = self.calc.run_lammps(atoms)
            e = (lammps_pe)/len(atoms) + row.data.tb_energy * self.tb_weight #+ E0 #energy per atom
            energy.append(e)
            fit_energy.append(row.data.total_energy)
            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            if average_distance>1.6:
                continue
            #tmp_rms = (e-(row.data.total_energy))
            #if self.fit_forces:
            #    total_forces = lammps_forces + row.data.tb_forces
            #    tmp_rms += np.linalg.norm(row.data.forces - total_forces)
            #rms.append(tmp_rms) #*sigma[i])
        print(energy)
        energy = np.array(energy) - np.min(energy)
        fit_energy = np.array(fit_energy) - np.min(fit_energy) 
        rms = np.linalg.norm(energy-fit_energy)
        wp = [str(p) for p in params]
        wp = " ".join(wp)
        with open(os.path.join(self.calc.output,self.output+"_rms.txt"),"a+") as f:
            f.write(str(rms)+" "+wp+"\n")
        return rms
    
    def fit(self,p0,bounds=None):
        '''
        bound all params = [0, np.inf]
        '''
        self.tb_weight = 1
        if self.optimizer_type=="Nelder-Mead":
            popt = scipy.optimize.minimize(self.objective,p0, method="Nelder-Mead",bounds=bounds)
        elif self.optimizer_type=="basinhopping":
            popt = scipy.optimize.basinhopping(self.objective,p0,niter=5,
                                               minimizer_kwargs={"method":"Nelder-Mead"},
                                               T=100,bounds=bounds)
        elif self.optimizer_type == "tb_weight":
            niter = 5
            for n in range(niter):
                if n<1:
                    continue
                tb_weight = (n+1)/niter
                self.tb_weight=tb_weight
                popt = scipy.optimize.minimize(self.objective,p0, method="Nelder-Mead",bounds=bounds)
                p0 = popt.x

        elif self.optimizer_type=="global":
            #fit each parameter individually, multiple times
            niter=2
            var_names = ['Q_CC' ,'alpha_CC', 'A_CC','BIJc_CC1', 'BIJc_CC2','BIJc_CC3','Beta_CC1', 'Beta_CC2', 'Beta_CC3']
            self.original_p0=p0.copy()
            for n in  range(niter):
                for i,p in enumerate(p0):
                    self.fit_param = i

                    popt = scipy.optimize.minimize(self.objective,p, method='Nelder-Mead')
                    self.original_p0[i] = pop        
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
    parser.add_argument('-oz','--optimizer_type',type=str,default="Nelder-Mead")
    args = parser.parse_args() 
   
    csfont = {'fontname':'serif',"size":18}

    if args.output==None:
        args.output = "fit_"+args.tbmodel+"_"+args.type+"_nkp"+args.nkp
    
    if int(args.nkp)==0:
        model = None
        intralayer_pot = "Rebo"
        interlayer_pot = "kolmogorov crespi"
    else:
        model = {"interlayer":"popov","intralayer":"porezag"}
        intralayer_pot = "Pz rebo"
        interlayer_pot = "Pz KC inspired"
    kd = np.sqrt(int(args.nkp))
    kmesh = (kd,kd,1)
    nkp = str(int(np.prod(kmesh)))
    
    model_dict = dict({"tight binding parameters":model,
                        "basis":"pz",
                        "kmesh":kmesh,
                        "parallel":"dask",
                        "intralayer potential":intralayer_pot,
                        "interlayer potential":interlayer_pot,
                        'output':args.output})
    if args.gendata=="True" and args.type=="interlayer":
   
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)

        print("assembling interlayer database")
        db = ase.db.connect('../data/bilayer_nkp'+nkp+'.db')
        df = pd.read_csv('../data/qmc.csv')
        for i, row in df.iterrows():
            print(i)
            atoms = get_bilayer_atoms(row['d'], row['disregistry'])
            if args.nkp > 0 :
                tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
            else:
                tb_energy,tb_forces = 0, np.zeros((len(atoms),3))
            db.write(atoms,data={"total_energy":row["energy"],'tb_energy':tb_energy/len(atoms)})

    if args.type=="interlayer" and args.fit=="True":
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
        print("fitting interlayer potential")
        db = ase.db.connect('../data/bilayer_nkp'+nkp+'.db')
        E0 = -154
        p0= [4.728912880179687, 32.40993806452906, -20.42597835994438,
             17.187123897218854, -23.370339868938927, 3.150121192047732,
             1.6724670937654809 ,13.646628785353208, 0.7907544823937784]
        potential = "KC inspired"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential,optimizer_type=args.optimizer_type)
        pfinal = fitting_obj.fit(p0)
        print(pfinal.x)

    if args.gendata=="True" and args.type=="intralayer":
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
        print("assembling intralayer database")
        subprocess.call('rm ../data/monolayer_nkp'+nkp+'.db',shell=True)
        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        file_list = glob.glob("../../../tBLG_DFT/grapheneCalc*",recursive=True)
        low_energy_dict={"total_energy":[],"atoms":[],"rebo_energy":[]}
        for f in file_list:
            print(os.path.join(f,"log"))
            try:
                atoms = ase.io.read(os.path.join(f,"log"),format="espresso-out")
                atoms.set_array('mol-id',np.ones(len(atoms)))
                total_energy = atoms.get_total_energy()

                print("natoms = ",len(atoms))
                print("total energy = ",total_energy)
                low_energy_dict["total_energy"].append(total_energy)
                low_energy_dict["atoms"].append(atoms)
                print("DFT success")
                #low_energy_dict["rebo_energy"].append(rebo_energy)
            except:
                print("DFT failed")
                continue


        ground_state = np.min(low_energy_dict["total_energy"])
        #ground_state_rebo = np.min(low_energy_dict["rebo_energy"])

        erange = .2 #eV/atom
        nn_range = (1.35,1.5)
        n_include = 0
        for i,a in enumerate(low_energy_dict["atoms"]):
            total_energy = low_energy_dict["total_energy"][i]
            pos = a.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            if average_distance < np.min(nn_range) or average_distance > np.max(nn_range):
                continue
            #print(low_energy_dict["total_energy"][i]-ground_state, low_energy_dict["rebo_energy"][i]-ground_state_rebo)
            if (total_energy - ground_state)/len(a) < erange:
                n_include +=1
                a.symbols = a.get_global_number_of_atoms() * "B"
                print(n_include, " energy (eV/atom) above gs = ",(total_energy - ground_state)/len(a))
                print("average distance ",average_distance,"\n")
                #try:
                plt.scatter(average_distance,(total_energy - ground_state)/len(a),c="blue")
                if int(args.nkp) > 0:

                    tb_energy,tb_forces = calc_obj.run_tight_binding(a)
                else:
                    tb_energy,tb_forces = 0, np.zeros((len(a),3))
                db.write(a,data={"total_energy":total_energy/len(a),'tb_forces':tb_forces,'tb_energy':tb_energy/len(a)})

                #except:
                #    print("failed Tight Binding")
                #    continue
        plt.savefig("dft_energies.png")
        plt.clf()
        
        model_dict = dict({"tight binding parameters":{"interlayer":"popov","intralayer":"porezag"},
                          "basis":"pz",
                          "kmesh":kmesh,
                          "parallel":"dask",
                          #"intralayer potential":os.path.join(args.output,"CH_pz.rebo_nkp225_final_version"),
                           "intralayer potential":"Pz rebo",
                           #"intralayer potential":"Rebo",
                          #"interlayer potential":os.path.join(args.output,"KC_insp_pz.txt_nkp225"),
                           "interlayer potential":"Pz KC inspired",
                           'output':args.output+"lat_con_test"
                          })
        
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)


        #get rid of configurations outside of applicability region
        p0 = [0.3134602960833/1.2, 4.7465390606595, 10953.544162170,\
             12388.79197798/1.1, 17.56740646509/1.1, 30.71493208065/1.1,\
             4.7204523127 , 1.4332132499, 1.3826912506]
        rebo_file = glob.glob(os.path.join(args.output+"lat_con_test","CH_pz.rebo*"),recursive=True)[0]
        write_rebo(p0,rebo_file)
        energy = []

        nconfig=0
        dft_min = 1e8
        for row in db.select():
            if row.data.total_energy<dft_min:
                dft_min = row.data.total_energy
        tegtb_energy = []
        dft_energy = []
        nn_dist = []
        tb_energy = []
        rebo_energy = []
        atoms_id =[]

        for row in db.select():

            atoms = db.get_atoms(id = row.id)
            atoms_id.append(row.id)
            atoms.calc = calc_obj
            lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
            e = (tote)/len(atoms) + row.data.tb_energy #energy per atom
            print("lammps energy = ",tote/len(atoms)," (eV/atom)")
            print("tb energy = ",row.data.tb_energy," (eV/atom)")
            #e = atoms.get_potential_energy()/len(atoms)
            tegtb_energy.append(e)
            dft_energy.append(row.data.total_energy)
            tb_energy.append(row.data.tb_energy)
            rebo_energy.append(tote/len(atoms))
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
        tb_min = tb_energy[rebo_min_ind]
        emprebo_min = rebo_energy[rebo_min_ind]
        rms = []
        for i,e in enumerate(tegtb_energy):
            line = np.linspace(0,1,10)
            ediff_line = line*((dft_energy[i]-dft_min) - (e-rebo_min)) + (e-rebo_min)
            tmp_rms = np.abs((dft_energy[i]-dft_min) - (e-rebo_min))
            if tmp_rms >0.15:
                print("deleting atom id = ",atoms_id[i])
                print("rms = ",tmp_rms)
                del db[atoms_id[i]]


    if args.type=="intralayer" and args.fit=="True":
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
        print("fitting intralayer potential")
        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        E0 = 0
        # 'Q_CC' ,'alpha_CC', 'A_CC'
        #'BIJc_CC1', 'BIJc_CC2','BIJc_CC3',
        #'Beta_CC1', 'Beta_CC2', 'Beta_CC3
        p0 = [0.3134602960833/1.2, 4.7465390606595, 10953.544162170,\
             12388.79197798/1.086, 17.56740646509/1.113, 30.71493208065/1.1,\
             4.7204523127 , 1.4332132499, 1.3826912506] 
        p0_bounds = [(-100,100),(-100,100),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]
        potential = "rebo"
        fitting_obj = fit_potentials_tblg(calc_obj, db, potential,optimizer_type=args.optimizer_type)
        pfinal = fitting_obj.fit(p0,bounds=p0_bounds)
        print(pfinal.x)

    if args.type=="interlayer" and args.test=="True":
       
        #intralayer_pot = glob.glob(os.path.join(args.output,"*CH_pz*"),recursive=True)[0]
        #model_dict["intralayer potential"] = intralayer_pot 
        #interlayer_pot = glob.glob(os.path.join(args.output,"KC_insp*_final_version"),recursive=True)[0]
        #model_dict["interlayer potential"] = interlayer_pot
        #interlayer_pot = glob.glob(os.path.join(args.output,"CC_QMC.KC"),recursive=True)[0]
        #model_dict["interlayer potential"] = interlayer_pot
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)

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
                if int(args.nkp)>0:
                    tb_energy,tb_forces = calc_obj.run_tight_binding(atoms)
                    total_energy = (atoms.get_potential_energy())/len(atoms) 
                else:
                    total_energy = (atoms.get_potential_energy())/len(atoms)
                    tb_energy=0
                tb_energy /= len(atoms)
                if total_energy<E0_tegt:
                    E0_tegt = total_energy
                qmc_total_energy = (row["energy"])

                energy_dis_tegt.append(total_energy)
                energy_dis_qmc.append(qmc_total_energy)
                energy_dis_tb.append(tb_energy)
                d_.append(row["d"])
                
            be_tegt, sep_tegt = get_binding_energy_sep(np.array(d_),np.array(energy_dis_tegt))
            #be_tegt = energy_dis_tegt[-1]
            #sep_tegt = 0
            be_qmc, sep_qmc = get_binding_energy_sep(np.array(d_),np.array(energy_dis_qmc))
            energy_min_tetb = get_energy_min(np.array(d_),np.array(energy_dis_tegt))
            energy_min_qmc = get_energy_min(np.array(d_),np.array(energy_dis_qmc))
            print(stacking+" TETB Energy Min = "+str(energy_min_tetb)+" (eV/atom)")
            print(stacking+" TETB layer separation = "+str(sep_tegt)+" (angstroms)")
            print(stacking+" qmc Energy Min = "+str(energy_min_qmc)+" (eV/atom)")
            print(stacking+" qmc layer separation = "+str(sep_qmc)+" (angstroms)")

            relative_tetb_energies.append(energy_dis_tegt)
            relative_qmc_energies.append(energy_dis_qmc)
            if int(args.nkp)>0:
                plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " TETB",c=colors[i])
            else:
                plt.plot(d_,np.array(energy_dis_tegt)-E0_tegt,label=stacking + " Classical",c=colors[i])
            #plt.scatter(d_,np.array(energy_dis_tb)-(energy_dis_tb[-1]),label=stacking + " TB",c=colors[i],marker=",")
            plt.scatter(d_,np.array(energy_dis_qmc)-E0_qmc,label=stacking + " qmc",c=colors[i])
        
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
        if int(args.nkp)>0:
            plt.title("TETB(nkp="+str(args.nkp)+")",**csfont)
        else:
            plt.title("Classical",**csfont)
        plt.tight_layout()
        plt.legend()
        plt.savefig("kc_insp_test_nkp"+str(args.nkp)+".jpg")
        plt.show()

        
    if args.type=="intralayer" and args.test=="True":
         
        #intralayer_pot = glob.glob(os.path.join(args.output,"CH_pz*_final_version"),recursive=True)[0]
        #model_dict["intralayer potential"] = intralayer_pot
        #interlayer_pot = glob.glob(os.path.join(args.output,"*KC_insp*"),recursive=True)[0]
        #model_dict["interlayer potential"] = interlayer_pot
        calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)

        model_dict = dict({"tight binding parameters":None,
                           "basis":"pz",
                           "intralayer potential":"Rebo",
                           "interlayer potential":"kolmogorov crespi",
                           'output':args.output +"classical"
                          })
        calc_obj_classical = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
        # 'Q_CC' ,'alpha_CC', 'A_CC'
        #'BIJc_CC1', 'BIJc_CC2','BIJc_CC3',
        #'Beta_CC1', 'Beta_CC2', 'Beta_CC3
        #p0 = [0.3134602960833/1.2, 4.7465390606595, 10953.544162170,\
        #     12388.79197798/1.086, 17.56740646509/1.113, 30.71493208065/1.1,\
        #     4.7204523127 , 1.4332132499, 1.3826912506]

        #rebo_file = glob.glob(os.path.join(args.output,"CH_pz.rebo*"),recursive=True)[0]
        #write_rebo(p0,rebo_file) 
        lat_con_test=False
        if lat_con_test:
            model_dict = dict({"tight binding parameters":{"interlayer":"popov","intralayer":"porezag"},
                          "basis":"pz",
                          "kmesh":kmesh,
                          "parallel":"dask",
                          "intralayer potential":"Pz rebo",
                          #"intralayer potential":"Rebo",
                          "interlayer potential":"Pz KC inspired",
                          'output':args.output +"lat_con_test"
                          })
            calc_obj = TETB_GRAPHENE_calc.TETB_GRAPHENE_Calc(model_dict)
            # 'Q_CC' ,'alpha_CC', 'A_CC'
            #'BIJc_CC1', 'BIJc_CC2','BIJc_CC3', 
            #'Beta_CC1', 'Beta_CC2', 'Beta_CC3'
            if args.nkp==1:
            #nkp = 1
                p0 = [0.3134602960833/1.2, 4.7465390606595, 10953.544162170,\
                 12388.79197798/1.1, 17.56740646509/1.1, 30.71493208065/1.1,\
                 4.7204523127 , 1.4332132499, 1.3826912506]
            else:
                
                p0 = [0.3134602960833/1.2, 4.7465390606595, 10953.544162170,\
                 12388.79197798/1.1, 17.56740646509/1.1, 30.71493208065/1.1,\
                 4.7204523127 , 1.4332132499, 1.3826912506]
            #rebo_file = glob.glob(os.path.join(args.output+"lat_con_test","CH_pz.rebo*"),recursive=True)[0]
            #write_rebo(p0,rebo_file)
            #write_rebo(p0,os.path.join(args.output+"lat_con_test","CH_pz.rebo_nkp225")) # CH_pz.rebo_nkp225

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
                atoms.calc = calc_obj
                total_energy = atoms.get_potential_energy()/len(atoms)
                #tb_energy_geom,tb_forces = calc_obj.run_tight_binding(atoms)
                #tb_energy[i] = tb_energy_geom/len(atoms)
                #lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
                #rebo_energy[i] = total_energy/len(atoms)
                #total_energy = tote + tb_energy_geom
                lat_con_energy[i] = total_energy
            fit_min_ind = np.argmin(lat_con_energy)
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
            print("DFT young's modulus = ",str(dft_params[0]))

            plt.plot(lat_con_list/np.sqrt(3),lat_con_energy-np.min(lat_con_energy),label = "rebo fit")
            #plt.plot(lat_con_list/np.sqrt(3),tb_energy-tb_energy[fit_min_ind],label = "tight binding energy")
            #plt.plot(lat_con_list/np.sqrt(3),rebo_energy - rebo_energy[fit_min_ind],label="rebo corrective energy")
            plt.plot(lat_con_list/np.sqrt(3), dft_energy-np.min(dft_energy),label="dft results")
            plt.xlabel(r"nearest neighbor distance ($\AA$)",**csfont)
            plt.ylabel("energy above ground state (eV/atom)",**csfont)
            plt.legend()
            plt.tight_layout()
            plt.savefig("rebo_lat_con_nkp"+str(nkp)+".jpg")
            plt.show()
            plt.clf()
            exit()
        

        """db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
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
        plt.show()"""
        
        db = ase.db.connect('../data/monolayer_nkp'+nkp+'.db')
        energy = []
        
        nconfig=0
        dft_min = 1e8
        for row in db.select():
            if row.data.total_energy<dft_min:
                dft_min = row.data.total_energy
        tegtb_energy = []
        dft_energy = []   
        nn_dist = []
        strain_x = []
        strain_y = []
        tb_energy = []
        rebo_energy = []
        atoms_id =[]
        unstrained_atoms = get_monolayer_atoms(0,0,a=2.462)
        unstrained_cell = unstrained_atoms.get_cell()
        
        for row in db.select():
    
            atoms = db.get_atoms(id = row.id)
            atoms_id.append(row.id)
            atoms.calc = calc_obj
            lammps_forces,lammps_pe,tote = calc_obj.run_lammps(atoms)
            if int(args.nkp)>0:
                e = (tote)/len(atoms) + row.data.tb_energy #energy per atom
                print("lammps energy = ",tote/len(atoms)," (eV/atom)")
                print("tb energy = ",row.data.tb_energy," (eV/atom)")
            else:
                e = (tote)/len(atoms)
            #e = atoms.get_potential_energy()/len(atoms)
            atoms.calc = calc_obj_classical
            classical_e = atoms.get_potential_energy()/len(atoms)
            tegtb_energy.append(e)
            print(row.data.total_energy)
            dft_energy.append(row.data.total_energy)
            tb_energy.append(row.data.tb_energy)
            rebo_energy.append(classical_e)
            nconfig+=1

            cell = atoms.get_cell()
            strain_x.append( np.linalg.norm(cell[0,:]))
            strain_y.append( np.linalg.norm(cell[1,:]))

            pos = atoms.positions
            distances = distance.cdist(pos, pos)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            average_distance = np.mean(min_distances)
            nn_dist.append(average_distance)
        dft_min = np.min(dft_energy)
        rebo_min_ind = np.argmin(tegtb_energy)
        rebo_min = tegtb_energy[rebo_min_ind]
        tb_min = tb_energy[rebo_min_ind]
        emprebo_min_ind = np.argmin(rebo_energy)
        emprebo_min = rebo_energy[emprebo_min_ind]
        unstrained_x = strain_x[rebo_min_ind]
        unstrained_y = strain_y[rebo_min_ind]
        rms_tetb  = []
        rms_rebo = []
        for i,e in enumerate(tegtb_energy):
            line = np.linspace(0,1,10)
            ediff_line = line*((dft_energy[i]-dft_min) - (e-rebo_min)) + (e-rebo_min)
            tmp_rms = np.linalg.norm((dft_energy[i]-dft_min) - (e-rebo_min))/(dft_energy[i]-dft_min)
            tmp_rebo_rms = np.linalg.norm((dft_energy[i]-dft_min) - (rebo_energy[i]-emprebo_min))/(dft_energy[i]-dft_min)
            #if tmp_rms >0.15:
            #    del db[atoms_id[i]]
            #    continue
            print("dft energy (eV/atom) = ",dft_energy[i]-dft_min)
            print("tegtb energy (eV/atom) = ",e-rebo_min)
            print("tb energy (eV/atom) = ",tb_energy[i]-tb_min)
            print("rebo correction energy (eV/atom) = ",rebo_energy[i]-emprebo_min)
            print("\n")
            average_distance = nn_dist[i]
            if nn_dist[i] > 1.5 or (dft_energy[i]-dft_min)>0.4:
                continue
            rms_tetb.append(tmp_rms)
            rms_rebo.append(tmp_rebo_rms)

            if i==0:
                plt.scatter(average_distance,e-rebo_min,color="red",label="TETB")
                #plt.scatter(average_distance,rebo_energy[i]-emprebo_min,color="orange",label="Rebo")
                #plt.scatter(average_disctance,tb_energyi[i]-tb_min,color="green",label="TB")
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue",label="DFT")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
            else:
                plt.scatter(average_distance,e-rebo_min,color="red")
                #plt.scatter(average_distance,rebo_energy[i]-emprebo_min,color="orange")
                #plt.scatter(average_distance,tb_energy[i]-tb_min,color="green")
                plt.scatter(average_distance,dft_energy[i]-dft_min,color="blue")
                plt.plot(average_distance*np.ones_like(line),ediff_line,color="black")
            
            #c = np.abs((dft_energy[i]-dft_min) - (e-rebo_min))
            #plt.scatter(average_distance,e-rebo_min,c=c,label="TETB")
        
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
        if int(args.nkp)>0:
            plt.title("TETB(nkp="+str(args.nkp)+")",**csfont)
        else:
            plt.title("Classical",**csfont)
        plt.legend()
        #plt.colorbar().set_label('RMS', rotation=270,**csfont)
        plt.clim((1e-5,1e-4))
        plt.tight_layout()
        plt.savefig("rebo_test_nkp"+str(nkp)+".jpg")
        plt.show()
        plt.clf()


        strain_x = (np.array(strain_x) - unstrained_x)/unstrained_x
        strain_y = (np.array(strain_y) - unstrained_y)/unstrained_y
        tegtb_energy = np.array(tegtb_energy)-rebo_min
        dft_energy = np.array(dft_energy)-dft_min
        rebo_energy = np.array(rebo_energy)-emprebo_min
        strain_x = np.round(strain_x,decimals=5)
        strain_x_set = np.unique(strain_x)
        for i,dx in enumerate(strain_x_set):
            strain_ind = np.isclose(strain_x,dx*np.ones_like(strain_x),rtol=1e-4)
            strain_sort_ind = np.argsort(strain_y[strain_ind])
            color = np.random.rand(3)
            plt.plot(strain_y[strain_ind][strain_sort_ind],tegtb_energy[strain_ind][strain_sort_ind],color="red")
            plt.scatter(strain_y[strain_ind][strain_sort_ind],dft_energy[strain_ind][strain_sort_ind],color="black")
            #plt.plot(strain_y[strain_ind][strain_sort_ind],dft_energy[strain_ind][strain_sort_ind],linestyle="dashed",c=color)
            #plt.plot(strain_y[strain_ind][strain_sort_ind],rebo_energy[strain_ind][strain_sort_ind],linestyle="dotted",c=color)
        plt.xlabel("strain along lattice vector 1")
        plt.ylabel("energy above ground state (eV/atom)")
        plt.title("Corrective Intralayer Potential for mLG with biaxial strain, num kpoints = "+str(args.nkp))
        plt.savefig("strain_energy_curves"+str(args.nkp)+".png")
        plt.clf()


        plt.scatter(strain_x,strain_y,c = ((np.array(tegtb_energy)) -(np.array(dft_energy))))
        plt.xlabel("strain along lattice vector 1")
        plt.ylabel("strain along lattice vector 2")
        plt.colorbar()
        plt.title("difference between TETB and dft total energies, nkp = "+str(args.nkp))
        plt.savefig("tetb_intralayer_surface_plot"+str(args.nkp)+".png")
        plt.clf()

        plt.scatter(strain_x,strain_y,c = ((np.array(rebo_energy)) -(np.array(dft_energy))))
        plt.xlabel("strain along lattice vector 1")
        plt.ylabel("strain along lattice vector 2")
        plt.colorbar()
        plt.title("difference between REBO and dft total energies")
        plt.savefig("rebo_intralayer_surface_plot.png")
        plt.clf()
            
