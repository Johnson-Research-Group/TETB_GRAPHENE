import ase.io
import numpy as np
import os
import json
import subprocess
from ase.calculators.calculator import Calculator, all_changes
from lammps import PyLammps
from lammps import lammps
import re
import glob

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

def init_pylammps_classical(atoms,kc_file = None,rebo_file = None):
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

    L.command("velocity	all create 0.0 87287 loop geom")
    # Interaction potential for carbon atoms
    ######################## Potential defition ########################

    L.command("pair_style       hybrid/overlay kolmogorov/crespi/full 10.0 0 rebo ")
    L.command("pair_coeff       * *   kolmogorov/crespi/full  "+kc_file+"    C") # long-range 
    L.command("pair_coeff      * * rebo "+rebo_file+" C ")

    ####################################################################
    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    return L

def init_pylammps_loop(atoms_dir,kc_file,rebo_file):
    N = len(glob.glob(os.path.join(atoms_dir,"tegt.data_*"),recursive=True))
    L = PyLammps(verbose=False)
    fname = atoms_dir+"/tegt.data_${i} "
    # LAMMPS input script for calculating potential energy of multiple structures
    L.command("units		metal")
    L.command("atom_style	full")
    L.command("atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    L.command("box tilt large")
    # Define variables
    L.command("variable i loop "+str(N))
    #L.command("variable filename equal "+atoms_dir+"/tegt.data_${i} ")
    #L.command("print \""+atoms_dir+"/tegt.data_${i}\" ")

    # Loop over structures
    L.command("label loop_start")

    # Read structure data
    L.command("read_data "+fname)

    # Initialize system (e.g., set potential, settings)
    L.command("group top type 1")
    L.command("mass 1 12.0100")

    #L.command("group bottom type 2")
    #L.command("mass 2 12.0100")

    L.command("velocity	all create 0.0 87287 loop geom")
    # Interaction potential for carbon atoms
    ######################## Potential defition ########################

    L.command("pair_style       hybrid/overlay reg/dep/poly 10.0 0 airebo 3")
    L.command("pair_coeff       * *   reg/dep/poly  "+kc_file+"   C C") # long-range 
    L.command("pair_coeff      * * airebo "+rebo_file+" C")

    ####################################################################
    L.command("timestep 0.00025")
    L.command("thermo 1")
    L.command("fix 1 all nve")
    L.command("run 0")
    
    # Output potential energy
    L.command("variable pe equal pe")
    L.command("print \"Potential energy of structure ${i} = ${pe}\"") #${filename}

    # Move to the next structure
    L.command("next i")
    L.command("jump SELF loop_start")
    return L

def init_lammps_loop(atoms_dir,kc_file,rebo_file):
    N = len(glob.glob(os.path.join(atoms_dir,"tegt.data_*"),recursive=True))
    fname = atoms_dir+"/tegt.data_${i} "
    input_file = "lammps_tetb.in"
    # LAMMPS input script for calculating potential energy of multiple structures
    with open(input_file,"w+") as f:
        f.write("variable i loop "+str(N)+"\n")
        #L.command("variable filename equal "+atoms_dir+"/tegt.data_${i} ")

        # Loop over structures
        f.write("label loop_start\n")

        f.write("units		metal\n")
        f.write("atom_style	full\n")
        f.write("atom_modify    sort 0 0.0\n")  # This is to avoid sorting the coordinates
        f.write("box tilt large\n")
        # Define variables
        
        # Read structure data
        f.write("read_data "+fname+"\n")

        # Initialize system (e.g., set potential, settings)
        f.write("group top type 1\n")
        f.write("mass 1 12.0100\n")

        #f.write("group bottom type 2\n")
        #f.write("mass 2 12.0100\n")

        f.write("velocity	all create 0.0 87287 loop geom\n")
        # Interaction potential for carbon atoms
        ######################## Potential defition ########################
        f.write("pair_style       airebo 3\n")
        #f.write("pair_style       hybrid/overlay reg/dep/poly 10.0 0 airebo 3\n")
        #f.write("pair_coeff       * *   reg/dep/poly  "+kc_file+"   C C\n") # long-range 
        #f.write("pair_coeff      * * airebo "+rebo_file+" C C\n")
        f.write("pair_coeff      * * "+rebo_file+" C\n")

        ####################################################################
        f.write("timestep 0.00025\n")
        f.write("thermo 1\n")
        f.write("fix 1 all nve\n")
        f.write("run 0\n")
        
        # Output potential energy
        f.write("variable pe equal pe\n")
        f.write("print \"Potential energy of structure ${i} = ${pe}\"\n") #${filename}
        f.write("clear\n")
        # Move to the next structure
        f.write("next i\n")
        f.write("jump SELF loop_start\n")
        
    lmp = lammps()
    lmp.file(input_file)

def run_lammps(atoms,kc_file,rebo_file,type="TETB"):
    """ evaluate corrective potential energy, forces in lammps 
    """
    
    if not atoms.has("mol-id"):
        mol_id = np.ones(len(atoms),dtype=np.int8)
        sym = atoms.get_chemical_symbols()
        top_layer_ind = np.where(np.array(sym)!=sym[0])[0]
        mol_id[top_layer_ind] += 1
        atoms.set_array("mol-id",mol_id)
    if type=="TETB":
        L = init_pylammps(atoms,kc_file=kc_file,rebo_file=rebo_file)
    elif type=="classical":
        L = init_pylammps_classical(atoms,kc_file=kc_file,rebo_file=rebo_file)
    forces = np.zeros((atoms.get_global_number_of_atoms(),3))

    L.run(0)
    pe = L.eval("pe")
    ke = L.eval("ke")
    for i in range(atoms.get_global_number_of_atoms()):
        forces[i,:] = L.atoms[i].force
    #del L

    return forces,pe,pe+ke

def get_pe(log_file):
    indices = []
    pe = []
    with open(log_file,"r") as f:
        lines = f.readlines()
        for l in lines:
            print("Potential energy of structure" in l)
            if "Potential energy of structure" in l:
                # Find all matches in the string
                numbers = re.findall(r'\d+\.?\d*', l)
                print(l)
                print(numbers)
                indices.append(numbers[-2])
                pe.append(numbers[-1])
    return np.array(pe)

def run_lammps_loop(atoms_dir,kc_file,rebo_file):

    L = init_lammps_loop(atoms_dir,kc_file=kc_file,rebo_file=rebo_file)

    #L.run(0)
    pe = get_pe("log.lammps")
    
    return pe

def write_kcinsp(params,kc_file):
    """write kc inspired potential """
    params = params[:9]
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['', "delta","C","C0 ","C2","C4","z0","A6","A8","A10"])
    with open(kc_file, 'w+') as f:
        f.write("# Refined parameters for Kolmogorov-Crespi Potential with taper function\n\
                #\n# "+headers+"         S     rcut\nC C "+params+" 1.0    2.0")
        
def write_kc(params,kc_file):
    """write kc inspired potential """
    
    params = " ".join([str(x) for x in params])
    headers = '               '.join(['','z0', 'C0', 'C2', 'C4', 'C', 'delta', 'lambda', 'A'])
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