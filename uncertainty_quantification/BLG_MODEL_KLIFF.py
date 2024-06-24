from typing import Dict, List, Optional, Tuple

import numpy as np
from kliff.dataset.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.models.parameter_transform import ParameterTransform
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress
from kliff.loss import Loss
from ase import Atoms
import subprocess
from BLG_ase_calc import BLG_classical
from datetime import datetime

class BLG_ClassicalComputeArguments(ComputeArguments):
    """
    KLIFF Total Energy Tight Binding Graphene potential computation functions.
    """

    implemented_property = ["energy"] #, "forces", "stress"]

    def __init__(
        self,
        conf: Configuration,
        supported_species: Dict[str, int],
        influence_distance: float,
        compute_energy: bool = True,
        compute_forces: bool = False,
        compute_stress: bool = False,
    ):
        if supported_species is None:
            species = sorted(set(conf.species))
            supported_species = {si: i for i, si in enumerate(species)}

            # using a single parameter for all species pairs
            self.specie_pairs_to_param_index = {
                (si, sj): 0 for si in species for sj in species
            }
        else:
            self.specie_pairs_to_param_index = (
                self._get_specie_pairs_to_param_index_map(
                    list(supported_species.keys())
                )
            )

        super(BLG_ClassicalComputeArguments, self).__init__(
            conf,
            supported_species,
            influence_distance,
            compute_energy,
            compute_forces,
            compute_stress,
        )

        """self.neigh = NeighborList(
            self.conf, influence_distance, padding_need_neigh=False
        )"""

    #Fix this function to work for TETB_slim.py
    def compute(self, params: Dict[str, Parameter]):
        coords = self.conf.coords
        species = self.conf.species
        cell = self.conf.cell

        if self.compute_energy:
            atoms = Atoms(positions=coords,cell=cell)
            param_list = np.squeeze(np.array([params[i].value for i in params]))[1:]
            self.output =  "classical_output_"+ str(hash(datetime.now()) )
            self.BLG_calc = BLG_classical(parameters = param_list,output = self.output)
            subprocess.call("rm -rf "+self.output,shell=True)
            self.results["energy"] = self.BLG_calc.get_total_energy(atoms) #energy
        """if self.compute_forces:
            forces = assemble_forces(
                forces_including_padding, len(coords), self.neigh.padding_image
            )
            self.results["forces"] = forces
        if self.compute_stress:
            volume = self.conf.get_volume()
            stress = assemble_stress(
                coords_including_padding, forces_including_padding, volume
            )
            self.results["stress"] = stress"""

    @staticmethod
    def _get_specie_pairs_to_param_index_map(
        species: List[str],
    ) -> Dict[Tuple[str, str], int]:
        """
        Return a map from a tuple of two species to the index of the corresponding
        parameter in the parameter array.

        For example, if the supported species are ["A", "B", "C"], then the map will be
        {(A, A): 0, (A, B): 1, (B, A): 1, (A, C): 2, (C, A): 2,
        (B, B): 3, (B, C): 4, (C, B): 4,
        (C, C): 5}.
        """
        n = len(species)

        speices_to_param_index_map = {}

        index = 0
        for i in range(n):
            si = species[i]
            for j in range(i, n):
                sj = species[j]
                speices_to_param_index_map[(si, sj)] = index
                if i != j:
                    speices_to_param_index_map[(sj, si)] = index

                index += 1

        return speices_to_param_index_map


class BLG_Classical(Model):
    """
    KLIFF built-in Lennard-Jones 6-12 potential model.

    This model supports multiple species, where a different set of parameters is used
    for each species pair. For example if species A, B, and C are provided, then there
    will be 6 values for each of the epsilon and sigma parameters. The order of the
    parameters is as follows: A-A, A-B, A-C, B-B, B-C, and C-C.

    Args:
        model_name: name of the model
        species: list of species. If None, there model will create a single value for
            each parameter, and all species pair will use the same parameters. If a
            list of species is provided, then the model will create a different set of
            parameters for each species pair.
        params_transform: parameter transform object. If None, no transformation is
            performed.
    """

    def __init__(
        self,
        model_name: str = "BLG_Classical",
        species: List[str] = None,
        params_transform: Optional[ParameterTransform] = None,
    ):
        self.species = species
        super(BLG_Classical, self).__init__(model_name, params_transform)

    def init_model_params(self):
        n = self._get_num_params()
        model_params = {"cutoff": Parameter(value=[10.0*0.529 for _ in range(n)])}
        
        self.rebo_params = np.array([0.14687637217609084,4.683462616941604,12433.64356176609,12466.479169306709,19.121905577450008,
                                     30.504342033258325,4.636516235627607,1.3641304165817836,1.3878198074813923])
        self.kc_params = np.array([3.379423382381699, 18.184672181803677, 13.394207130830571, 0.003559135312169, 6.074935002291668,
                                        0.719345289329483, 3.293082477932360, 13.906782892134125])
        
        for i in range(len(self.rebo_params)):
            model_params.update({"rebo_"+str(i):Parameter(value=[self.rebo_params[i] for _ in range(n)])})

        for i in range(len(self.kc_params)):
            model_params.update({"kc_"+str(i):Parameter(value=[self.kc_params[i] for _ in range(n)])})

        return model_params

    def init_influence_distance(self):
        return self.model_params["cutoff"][0]

    def init_supported_species(self):
        if self.species is None:
            return None
        else:
            return {s: i for i, s in enumerate(self.species)}

    def get_compute_argument_class(self):
        return BLG_ClassicalComputeArguments

    def _get_num_params(self):
        if self.species is None:
            n = 1
        else:
            n = len(self.species)

        return (n + 1) * n // 2