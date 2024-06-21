from typing import Dict, List, Optional, Tuple

import numpy as np
from TETB_slim import TETB_slim
from kliff.dataset.dataset import Configuration
from kliff.models.model import ComputeArguments, Model
from kliff.models.parameter import Parameter
from kliff.models.parameter_transform import ParameterTransform
from kliff.neighbor import NeighborList, assemble_forces, assemble_stress
from kliff.loss import Loss
from ase import Atoms


class TETBComputeArguments(ComputeArguments):
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

        super(TETBComputeArguments, self).__init__(
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
            self.results["energy"] = self.tetb_calc.get_total_energy(atoms) #energy
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


class TETB_KLIFF_Model(Model):
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
        model_name: str = "TETB-GRAPHENE",
        species: List[str] = None,
        params_transform: Optional[ParameterTransform] = None,
    ):
        self.species = species
        super(TETB_KLIFF_Model, self).__init__(model_name, params_transform)

    def init_model_params(self):
        n = self._get_num_params()
        model_params = {"cutoff": Parameter(value=[10.0*0.529 for _ in range(n)])}
        
        self.rebo_params = np.array([0.34563531369329037,4.6244265008884184,11865.392552302139,14522.273379352482,7.855493960028371,
                                     40.609282094464604,4.62769509546907,0.7945927858501145,2.2242248220983427])
        self.kc_params = np.array([16.34956726725497, 86.0913106836395, 66.90833163067475, 24.51352633628406, -103.18388323245665,
                                   1.8220964068356134, -2.537215908290726, 18.177497643244706, 2.762780721646056])
        self.interlayer_hopping_params = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,-0.0978079, 0.0577363, -0.0262833, 0.0094388,
                                                   -0.0024695, 0.0003863, -0.3969243, 0.3477657, -0.2357499, 0.1257478,-0.0535682, 0.0181983,
                                                     -0.0046855, 0.0007303,0.0000225, -0.0000393])
        self.intralayer_hopping_params = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,-0.0673216, 0.0316900, -0.0117293, 0.0033519,
                                                    -0.0004838, -0.0000906,-0.3793837, 0.3204470, -0.1956799, 0.0883986,-0.0300733, 0.0074465,
                                                    -0.0008563, -0.0004453, 0.0003842, -0.0001855])
        
        for i in range(len(self.rebo_params)):
            model_params.update({"rebo_"+str(i):Parameter(value=[self.rebo_params[i] for _ in range(n)])})

        for i in range(len(self.kc_params)):
            model_params.update({"kc_"+str(i):Parameter(value=[self.kc_params[i] for _ in range(n)])})

        for i in range(len(self.interlayer_hopping_params)):
            model_params.update({"interlayer_hopping_params_"+str(i):Parameter(value=[self.interlayer_hopping_params[i] for _ in range(n)])})

        for i in range(len(self.intralayer_hopping_params)):
            model_params.update({"intralayer_hopping_params_"+str(i):Parameter(value=[self.intralayer_hopping_params[i] for _ in range(n)])})
        

        return model_params

    def init_influence_distance(self):
        return self.model_params["cutoff"][0]

    def init_supported_species(self):
        if self.species is None:
            return None
        else:
            return {s: i for i, s in enumerate(self.species)}

    def get_compute_argument_class(self):
        return TETBComputeArguments

    def _get_num_params(self):
        if self.species is None:
            n = 1
        else:
            n = len(self.species)

        return (n + 1) * n // 2
    
    
    if __name__ == "__main__":
        model = TETB_KLIFF()
        model.echo_model_params()

        #fitting parameters
        model.set_opt_params(sigma=[["default"]], epsilon=[["default"]])
        model.echo_opt_params()

        calc = Calculator(model)
        calc.create(configs)

        # loss
        loss = Loss(calc, nprocs=1)
        result = loss.minimize(method="L-BFGS-B", options={"disp": True, "maxiter": 10})


        # print optimized parameters
        model.echo_opt_params()
        model.save("kliff_model.yaml")