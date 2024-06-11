import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps

def init_pylammps(atoms,kc_file = None,rebo_file = None):
    """ create pylammps object and calculate corrective potential energy 
    """
    ntypes = len(set(atoms.get_chemical_symbols()))
    data_file = "tegt.data"
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
    
    if ntypes ==2:
        L.command("pair_style       hybrid/overlay reg/dep/poly 10.0 0 airebo 3")
        L.command("pair_coeff       * *   reg/dep/poly  "+kc_file+"   C C") # long-range 
        L.command("pair_coeff      * * airebo "+rebo_file+" C C")

    else:
        L.command("pair_style       airebo 3")
        L.command("pair_coeff      * * "+rebo_file+" C")

    ####################################################################
    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    return L

def run_lammps(atoms,kc_file,rebo_file):
    """ evaluate corrective potential energy, forces in lammps 
    """
    
    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)

    L = init_pylammps(atoms,kc_file=kc_file,rebo_file=rebo_file)
    forces = np.zeros((atoms.get_global_number_of_atoms(),3))

    pe = L.eval("pe")
    ke = L.eval("ke")
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force
    del L

    return forces,pe,pe+ke

def write_kcinsp(params,kc_file):
    """write kc inspired potential """
    params = params[:9]
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
    

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
                nl = str(np.squeeze(param_dict[line_key]))+" "+line_key+" \n"
                new_lines.append(nl)
            else:
                new_lines.append(l)
    with open(rebo_file, 'w') as f:        
        f.writelines(new_lines)