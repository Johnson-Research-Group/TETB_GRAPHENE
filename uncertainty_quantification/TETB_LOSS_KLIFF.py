import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.optimize
from loguru import logger
from TB_Utils import *
from kliff import parallel
from kliff.calculators.calculator import Calculator, _WrapperCalculator
from kliff.calculators.calculator_torch import CalculatorTorch
from kliff.dataset.weight import Weight
from kliff.error import report_import_error
from datetime import datetime
from TETB_slim import TETB_slim
import subprocess
import time

try:
    import torch

    torch_avail = True
except ImportError:
    torch_avail = False

try:
    from mpi4py import MPI

    mpi4py_avail = True
except ImportError:
    mpi4py_avail = False

try:
    from geodesicLM import geodesiclm

    geodesicLM_avail = True
except ImportError:
    geodesicLM_avail = False

def energy_forces_residual(
    identifier: str,
    natoms: int,
    weight: Weight,
    prediction: np.array,
    reference: np.array,
    data: Dict[str, Any],
):
    """
    A residual function using both energy and forces.

    The residual is computed as

    .. code-block::

       weight.config_weight * wi * (prediction - reference)

    where ``wi`` can be ``weight.energy_weight`` or ``weight.forces_weight``, depending
    on the property.

    Args:
        identifier: (unique) identifier of the configuration for which to compute the
            residual. This is useful when you want to weigh some configuration
            differently.
        natoms: number of atoms in the configuration
        weight: an instance that computes the weight of the configuration in the loss
            function.
        prediction: prediction computed by calculator, 1D array
        reference: references data for the prediction, 1D array
        data: additional data for calculating the residual. Supported key value
            pairs are:
            - normalize_by_atoms: bool (default: True)
            If ``normalize_by_atoms`` is ``True``, the residual is divided by the number
            of atoms in the configuration.

    Returns:
        1D array of the residual

    Note:
        The length of `prediction` and `reference` (call it `S`) are the same, and it
        depends on `use_energy` and `use_forces` in Calculator. Assume the
        configuration contains of `N` atoms.

        1. If `use_energy == True` and `use_forces == False`, then `S = 1`.
        `prediction[0]` is the potential energy computed by the calculator, and
        `reference[0]` is the reference energy.

        2. If `use_energy == False` and `use_forces == True`, then `S = 3N`.
        `prediction[3*i+0]`, `prediction[3*i+1]`, and `prediction[3*i+2]` are the
        x, y, and z component of the forces on atom i in the configuration, respectively.
        Correspondingly, `reference` is the 3N concatenated reference forces.

        3. If `use_energy == True` and `use_forces == True`, then `S = 3N + 1`.
        `prediction[0]` is the potential energy computed by the calculator, and
        `reference[0]` is the reference energy.
        `prediction[3*i+1]`, `prediction[3*i+2]`, and `prediction[3*i+3]` are the
        x, y, and z component of the forces on atom i in the configuration, respectively.
        Correspondingly, `reference` is the 3N concatenated reference forces.
    """

    # extract up the weight information
    config_weight = weight.config_weight
    energy_weight = weight.energy_weight
    forces_weight = weight.forces_weight

    # obtain residual and properly normalize it
    residual = config_weight * (prediction - reference)
    residual[0] *= energy_weight
    residual[1:] *= forces_weight

    if data["normalize_by_natoms"]:
        residual /= natoms

    return residual


def energy_residual(
    identifier: str,
    natoms: int,
    weight: Weight,
    prediction: np.array,
    reference: np.array,
    data: Dict[str, Any],
):
    """
    A residual function using just the energy.

    See the documentation of :meth:`energy_forces_residual` for the meaning of the
    arguments.
    """

    # extract up the weight information
    config_weight = weight.config_weight
    energy_weight = weight.energy_weight

    # obtain residual and properly normalize it
    residual = config_weight * energy_weight * (prediction - reference)

    if data["normalize_by_natoms"]:
        residual /= natoms

    return residual


def forces_residual(
    identifier: str,
    natoms: int,
    weight: Weight,
    prediction: np.array,
    reference: np.array,
    data: Dict[str, Any],
):
    """
    A residual function using just the forces.

    See the documentation of :meth:`energy_forces_residual` for the meaning of the
    arguments.
    """

    # extract up the weight information
    config_weight = weight.config_weight
    forces_weight = weight.forces_weight

    # obtain residual and properly normalize it
    residual = config_weight * forces_weight * (prediction - reference)

    if data["normalize_by_natoms"]:
        residual /= natoms

    return residual

        
class LossTETBModel:
    """
    Loss function class to optimize the physics-based potential parameters.

    Args:
        calculator: Calculator to compute prediction from atomic configuration using
            a potential model.
        nprocs: Number of processes to use..
        residual_fn: function to compute residual, e.g. :meth:`energy_forces_residual`,
            :meth:`energy_residual`, and :meth:`forces_residual`. See the documentation
            of :meth:`energy_forces_residual` for the signature of the function.
            Default to :meth:`energy_forces_residual`.
        residual_data: data passed to ``residual_fn``; can be used to fine tune the
            residual function. Default to
            {
                "normalize_by_natoms": True,
            }
            See the documentation of :meth:`energy_forces_residual` for more.
    """

    scipy_minimize_methods = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]
    scipy_minimize_methods_not_supported_args = ["bounds"]
    scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
    scipy_least_squares_methods_not_supported_args = ["bounds"]

    def __init__(
        self,
        calculator: Calculator,
        nprocs: int = 1,
        residual_fn: Optional[Callable] = None,
        residual_data: Optional[Dict[str, Any]] = None,
        hopping_data:Optional[Dict[str,Any]] = None,
        loss_type="all"
    ):
        default_residual_data = {
            "normalize_by_natoms": True,
        }

        residual_data = _check_residual_data(residual_data, default_residual_data)

        self.calculator = calculator
        self.nprocs = nprocs

        self.residual_data = residual_data
        self.hopping_data = hopping_data
        self.loss_type = loss_type
        if self.hopping_data is not None:
            self.interlayer_hopping_data = hopping_data["interlayer"]
            self.intralayer_hopping_data = hopping_data["intralayer"]

        if residual_fn is None:
            if isinstance(self.calculator, _WrapperCalculator):
                self.calc_list = self.calculator.get_calculator_list()
                self.residual_fn = []
                for calculator in self.calc_list:
                    if calculator.use_energy and calculator.use_forces:
                        residual_fn = energy_forces_residual
                    elif calculator.use_energy:
                        residual_fn = energy_residual
                    elif calculator.use_forces:
                        residual_fn = forces_residual
                    else:
                        raise RuntimeError("Calculator does not use energy or forces.")
                    self.residual_fn.append(residual_fn)
            else:
                if calculator.use_energy and calculator.use_forces:
                    residual_fn = energy_forces_residual
                elif calculator.use_energy:
                    residual_fn = energy_residual
                elif calculator.use_forces:
                    residual_fn = forces_residual
                else:
                    raise RuntimeError("Calculator does not use energy or forces.")
                self.residual_fn = residual_fn
        else:
            # TODO this will not work for _WrapperCalculator
            self.residual_fn = residual_fn

        logger.debug(f"`{self.__class__.__name__}` instantiated.")

    def minimize(self, method: str = "L-BFGS-B", **kwargs):
        """
        Minimize the loss.

        Args:
            method: minimization methods as specified at:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

            kwargs: extra keyword arguments that can be used by the scipy optimizer
        """
        kwargs = self._adjust_kwargs(method, **kwargs)

        logger.info(f"Start minimization using method: {method}.")
        result = self._scipy_optimize(method, **kwargs)
        logger.info(f"Finish minimization using method: {method}.")

        # update final optimized parameters
        self.calculator.update_model_params(result.x)

        return result

    def _adjust_kwargs(self, method, **kwargs):
        """
        Check kwargs and adjust them as necessary.
        """

        if method in self.scipy_least_squares_methods:
            # check support status
            for i in self.scipy_least_squares_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        f"Argument `{i}` should not be set via the `minimize` method. "
                        "It it set internally."
                    )

            # adjust bounds
            if self.calculator.has_opt_params_bounds():
                if method in ["trf", "dogbox"]:
                    bounds = self.calculator.get_opt_params_bounds()
                    lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
                    ub = [b[1] if b[1] is not None else np.inf for b in bounds]
                    bounds = (lb, ub)
                    kwargs["bounds"] = bounds
                else:
                    raise LossError(f"Method `{method}` cannot handle bounds.")

        elif method in self.scipy_minimize_methods:
            # check support status
            for i in self.scipy_minimize_methods_not_supported_args:
                if i in kwargs:
                    raise LossError(
                        f"Argument `{i}` should not be set via the `minimize` method. "
                        "It it set internally."
                    )

            # adjust bounds
            if isinstance(self.calculator, _WrapperCalculator):
                calculators = self.calculator.calculators
            else:
                calculators = [self.calculator]
            for calc in calculators:
                if calc.has_opt_params_bounds():
                    if method in ["L-BFGS-B", "TNC", "SLSQP"]:
                        bounds = self.calculator.get_opt_params_bounds()
                        kwargs["bounds"] = bounds
                    else:
                        raise LossError(f"Method `{method}` cannot handle bounds.")
        else:
            raise LossError(f"Minimization method `{method}` not supported.")

        return kwargs

    def _scipy_optimize(self, method, **kwargs):
        """
        Minimize the loss use scipy.optimize.least_squares or scipy.optimize.minimize
        methods. A user should not call this function, but should call the ``minimize``
        method.
        """

        size = parallel.get_MPI_world_size()

        if size > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            logger.info(f"Running in MPI mode with {size} processes.")

            if self.nprocs > 1:
                logger.warning(
                    f"Argument `nprocs = {self.nprocs}` provided at initialization is "
                    f"ignored. When running in MPI mode, the number of processes "
                    f"provided along with the `mpiexec` (or `mpirun`) command is used."
                )

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                # geodesic LM
                if method == "geodesiclm":
                    if not geodesicLM_avail:
                        report_import_error("geodesiclm")
                    else:
                        minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares
                func = self._get_residual_MPI

            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self._get_loss_MPI

            if rank == 0:
                result = minimize_fn(func, x, method=method, **kwargs)
                # notify other process to break func
                break_flag = True
                for i in range(1, size):
                    comm.send(break_flag, dest=i, tag=i)
            else:
                func(x)
                result = None

            result = comm.bcast(result, root=0)

            return result

        else:
            # 1. running MPI with 1 process
            # 2. running without MPI at all
            # both cases are regarded as running without MPI

            if self.nprocs == 1:
                logger.info("Running in serial mode.")
            else:
                logger.info(
                    f"Running in multiprocessing mode with {self.nprocs} processes."
                )

                # Maybe one thinks he is using MPI because nprocs is used
                if mpi4py_avail:
                    logger.warning(
                        "`mpi4py` detected. If you try to run in MPI mode, you should "
                        "execute your code via `mpiexec` (or `mpirun`). If not, ignore "
                        "this message."
                    )

            x = self.calculator.get_opt_params()
            if method in self.scipy_least_squares_methods:
                if method == "geodesiclm":
                    if not geodesicLM_avail:
                        report_import_error("geodesiclm")
                    else:
                        minimize_fn = geodesiclm
                else:
                    minimize_fn = scipy.optimize.least_squares

                func = self._get_residual
            elif method in self.scipy_minimize_methods:
                minimize_fn = scipy.optimize.minimize
                func = self._get_loss

            result = minimize_fn(func, x, method=method, **kwargs)
            return result

    def _get_residual(self, x):
        """
        Compute the residual in serial or multiprocessing mode.

        This is a callable for optimizing method in scipy.optimize.least_squares,
        which is passed as the first positional argument.

        Args:
            x: optimizing parameter values, 1D array
        """

        # publish params x to predictor
        self.calculator.update_model_params(x)

        cas = self.calculator.get_compute_arguments()

        # TODO the if else could be combined
        if isinstance(self.calculator, _WrapperCalculator):
            X = zip(cas, self.calc_list, self.residual_fn)
            if self.nprocs > 1:
                residuals = parallel.parmap2(
                    self._get_residual_single_config,
                    X,
                    self.residual_data,
                    nprocs=self.nprocs,
                    tuple_X=True
                )
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca, calculator, residual_fn in X:
                    current_residual = self._get_residual_single_config(
                        ca, calculator, residual_fn, self.residual_data
                    )
                    residual = np.concatenate((residual, current_residual))

        else:
            if self.nprocs > 1:
                residuals = parallel.parmap2(
                    self._get_residual_single_config,
                    cas,
                    self.calculator,
                    self.residual_fn,
                    self.residual_data,
                    nprocs=self.nprocs,
                    tuple_X=False,
                )
                residual = np.concatenate(residuals)
            else:
                residual = []
                for ca in cas:
                    current_residual = self._get_residual_single_config(
                        ca, self.calculator, self.residual_fn, self.residual_data
                    )
                    residual = np.concatenate((residual, current_residual))

        return residual

    def _get_loss(self, x):
        """
        Compute the loss in serial or multiprocessing mode.

        This is a callable for optimizing method in scipy.optimize.minimize,
        which is passed as the first positional argument.

        Args:
            x: 1D array, optimizing parameter values
        """
        start = time.time()
        #only write parameter files once per set of parameters
        self.output = "TETB_output_"+ str(hash(datetime.now()) )
        if self.loss_type=="interlayer":
            opt_params = self.calculator.model.get_model_params()
            param_list = np.squeeze(np.array([opt_params[i].value for i in opt_params]))[1:]
            param_list[9:18] = x
        elif self.loss_type=="intralayer":
            opt_params = self.calculator.model.get_model_params()
            param_list = np.squeeze(np.array([opt_params[i].value for i in opt_params]))[1:]
            param_list[:9] = x
        else:
            param_list = x

        calc = TETB_slim(parameters=param_list,output = self.output)
        self.tetb_calc = calc
        residual = self._get_residual(x)
        energy_loss = 0.5 * np.linalg.norm(residual) ** 2
        subprocess.call("rm -rf "+self.output,shell=True)
        #these evaluations of hopping parameterizations are very quick

        if self.hopping_data is not None:
            interlayer_hopping_params = x[18:38]
            intralayer_hopping_params = x[38:]
            hoppings = popov_hopping(self.interlayer_hopping_data["disp"],params=np.vstack((interlayer_hopping_params[:10],interlayer_hopping_params[10:])))
            hopping_rmse_array = np.linalg.norm((hoppings-self.interlayer_hopping_data["hopping"]))/self.interlayer_hopping_data["hopping"]

            hoppings = porezag_hopping(self.intralayer_hopping_data["disp"],params=np.vstack((intralayer_hopping_params[:10],intralayer_hopping_params[10:])))
            hopping_rmse_array = np.append(hopping_rmse_array,np.linalg.norm((hoppings-self.intralayer_hopping_data["hopping"]))/self.intralayer_hopping_data["hopping"])
        else:
            hopping_rmse_array = 0
        loss = 0.5 * energy_loss + 0.5 * np.linalg.norm(hopping_rmse_array)
        end = time.time()
        print("time for loss function  = ",end-start)

        """with open("mcmc_parameters.txt","a+") as f:
            ensemble_str = " ".join([str(i) for i in x])
            f.write(ensemble_str+"\n")"""
        return loss

    def _get_residual_MPI(self, x):
        def residual_my_chunk(x):
            # broadcast parameters
            x = comm.bcast(x, root=0)
            # publish params x to predictor
            self.calculator.update_model_params(x)

            residual = []
            for ca in cas:
                current_residual = self._get_residual_single_config(
                    ca, self.calculator, self.residual_fn, self.residual_data
                )
                residual.extend(current_residual)
            return residual

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # get my chunk of data
        cas = self._split_data()

        while True:
            if rank == 0:
                break_flag = False
                for i in range(1, size):
                    comm.send(break_flag, dest=i, tag=i)
                residual = residual_my_chunk(x)
                all_residuals = comm.gather(residual, root=0)
                return np.concatenate(all_residuals)
            else:
                break_flag = comm.recv(source=0, tag=rank)
                if break_flag:
                    break
                else:
                    residual = residual_my_chunk(x)
                    all_residuals = comm.gather(residual, root=0)

    def _get_loss_MPI(self, x):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        residual = self._get_residual_MPI(x)
        if rank == 0:
            loss = 0.5 * np.linalg.norm(residual) ** 2
        else:
            loss = None

        return loss

    # NOTE this function can be called only once, no need to call it each time
    # _get_residual_MPI is called
    def _split_data(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # get a portion of data based on rank
        cas = self.calculator.get_compute_arguments()
        # random.shuffle(cas)

        rank_size = len(cas) // size
        # last rank deal with the case where len(cas) cannot evenly divide size
        if rank == size - 1:
            cas = cas[rank_size * rank :]
        else:
            cas = cas[rank_size * rank : rank_size * (rank + 1)]

        return cas

    #@staticmethod
    def _get_residual_single_config(self,ca, calculator, residual_fn, residual_data):
        # prediction data
        ca.tetb_calc = self.tetb_calc
        calculator.compute(ca)
        pred = calculator.get_prediction(ca)

        # reference data
        ref = calculator.get_reference(ca)

        conf = ca.conf
        identifier = conf.identifier
        weight = conf.weight
        natoms = conf.get_num_atoms()

        residual = residual_fn(identifier, natoms, weight, pred, ref, residual_data)

        return residual
    
def _check_residual_data(data: Dict[str, Any], default: Dict[str, Any]):
    """
    Check whether user provided residual data is valid, and add default values if not
    provided.
    """
    if data is not None:
        for key, value in data.items():
            if key not in default:
                raise LossError(
                    f"Expect the keys of `residual_data` to be one or combinations of "
                    f"{', '.join(default.keys())}; got {key}. "
                )
            else:
                default[key] = value

    return default

class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg

