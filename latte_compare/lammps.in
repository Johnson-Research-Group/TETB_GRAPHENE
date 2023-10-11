# To be used with the latte-lib input file.  

units		metal
atom_style	full
atom_modify    sort 0 0.0  # This is to avoid sorting the coordinates
box tilt large

read_data tegt.data
group top type 1
group bottom type 2

mass 1 12.0100
mass 2 12.0200

velocity	all create 0.0 87287 loop geom
# Interaction potential for carbon atoms
######################## Potential defition ########################
pair_style zero 5.0 nocoeff
pair_coeff * *
####################################################################


timestep 0.00025

variable latteE equal "(ke + f_2)"
variable kinE equal "ke"
variable potE equal "f_2"

thermo 1
thermo_style   custom step pe ke etotal temp v_latteE
log log.test
fix		1 all nve

fix   2 all latte NULL
dump           mydump2 all custom 1 dump.tegt_latte id type x y z fx fy fz

run 0
