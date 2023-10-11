#include("LAMMPS.jl")
using LAMMPS
function CorrectivePotential(data_file)
    #includes Kolmogorov Cresip inspired potential, and rebo
    lmp = LMP()
    command(lmp, "units		metal")
    command(lmp, "atom_style	full")
    command(lmp, "atom_modify    sort 0 0.0")  # This is to avoid sorting the coordinates
    command(lmp, "box tilt large")

    command(lmp, " read_data "*data_file)
    command(lmp, "group top type 1")
    command(lmp, "group bottom type 2")

    command(lmp, "mass 1 12.0100")
    command(lmp, "mass 2 12.0200")

    command(lmp, "velocity	all create 0.0 87287 loop geom")
    # Interaction potential for carbon atoms
    ######################## Potential defition ########################

    command(lmp, "pair_style       hybrid/overlay reg/dep/poly 10.0 0 rebo")
    #command(lmp, "pair_coeff	1 1 rebo CH_pz.rebo C C")
    command(lmp, "pair_coeff       * *   reg/dep/poly  KC_insp_pz.txt   C C") # long-range 
    command(lmp, "pair_coeff      * * rebo CH_pz.rebo C C")
    ####################################################################


    command(lmp, "timestep 0.00025")

    command(lmp, "compute 0 all pair reg/dep/poly")
    command(lmp, "variable Evdw  equal c_0[1]")
    command(lmp, "variable Erep  equal c_0[2]")

    command(lmp, "thermo 1")
    command(lmp, "thermo_style   custom step pe ke etotal temp v_Erep v_Evdw")
    command(lmp, "fix 1 all nve")
    command(lmp, "run 0")

    # extract output
    
    forces = extract_atom(lmp, "f")
    energies = extract_compute(lmp, "pot_e", LAMMPS.API.LMP_STYLE_GLOBAL, LAMMPS.API.LMP_TYPE_SCALAR)
    return forces,energies
end
