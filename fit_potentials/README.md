Fit potentials for TETB models. DFT/QMC fitting Data is stored in TETB_GRAPHENE/data. data/monolayer_nkp(N).db is an ASE database that consists of 50 biaxial strained monolayer graphene configurations with DFT Total Energy data with key="total_energy", and tight binding energy with key="tb_energy" calculated with N kpoints. data/bilayer_nkp(N).db is an ASE database that consists of 40 bilayer graphene configurations at different interlayer separations and stackings with QMC Total Energy data with key="total_energy", and tight binding energy with key="tb_energy" calculated with N kpoints. 

In order to fit a potential, run the following python command:

```python -m (tight binding model) -t (interlayer/intralayer) -g (generate database) -f (fit potential) -s (test potential) -k (number of kpoints) -o (output)```

for example if one wanted to fit the interlayer residual potential for a TETB model with 121 kpoints, with the tight binding energy calculated with the Popov Van Alsenoy tight binding model:

```python -m popov -t interlayer -g True -f True -k 121 -o interlayer_residual_nkp121```  
