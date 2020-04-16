# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061

"""

import sys
import os
sys.path.append(os.path.abspath("DensityMatrixReconstruction/src/"))
sys.path.append('../src/')
import utils
#from qiskit.aqua.utils import utils
from mpi4py import MPI
import netket as nk

from typing import Optional, List, Callable, Union
import logging
import warnings
from time import time

import numpy as np
from qiskit import ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit import BasicAer
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.operators import (OperatorBase, ExpectationBase, CircuitStateFn,
                                   LegacyBaseOperator, ListOp, I, PauliExpectation, PauliOp, MatrixExpectation, PauliBasisChange, CircuitSampler)
from qiskit.aqua.operators.legacy import (MatrixOperator, WeightedPauliOperator,
                                          TPBGroupedWeightedPauliOperator)

from qiskit.aqua.components.optimizers import Optimizer, SLSQP
from qiskit.aqua.components.variational_forms import VariationalForm, RY
from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.utils.backend_utils import is_aer_provider
from qiskit.quantum_info.operators.pauli import Pauli
from ..vq_algorithm import VQAlgorithm, VQResult
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult




### DNN additions ###
#import utils
###


logger = logging.getLogger(__name__)

# disable check for var_forms, optimizer setter because of pylint bug
# pylint: disable=no-member


class VQE(VQAlgorithm, MinimumEigensolver):
    r"""
    The Variational Quantum Eigensolver algorithm.

    `VQE <https://arxiv.org/abs/1304.3061>`__ is a hybrid algorithm that uses a
    variational technique and interleaves quantum and classical computations in order to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.

    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (ansatz) from Aqua's :mod:`~qiskit.aqua.components.variational_forms`, and one
    of the classical :mod:`~qiskit.aqua.components.optimizers`. The ansatz is varied, via its set
    of parameters, by the optimizer, such that it works towards a state, as determined by the
    parameters applied to the variational form, that will result in the minimum expectation value
    being measured of the input operator (Hamiltonian).

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  Aqua provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the variational form being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the variational form for a preferred value, based on its
    given initial state. If the variational form returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the variational form provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the variational form returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.
    """

    def __init__(self,
                 operator: Optional[OperatorBase] = None,
                 var_form: Optional[VariationalForm] = None,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 expectation_value: Optional[ExpectationBase] = None,
                 max_evals_grouped: int = 1,
                 aux_operators: Optional[OperatorBase] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 # TODO delete all instances of auto_conversion
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
                 rbm_params: None = None) -> None:
        """

        Args:
            operator: Qubit operator of the Observable
            var_form: A parameterized variational form (ansatz).
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            expectation_value: expectation value
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            aux_operators: Optional ListOp or list of auxiliary operators to be evaluated with the
                eigenstate of the minimum eigenvalue main result and their expectation values
                returned. For instance in chemistry these can be dipole operators, total particle
                count operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.`
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('max_evals_grouped', max_evals_grouped, 1)
        # TODO delete all instances of self._use_simulator_snapshot_mode
        self._use_simulator_snapshot_mode = False
        # TODO delete instances of self._auto_conversion
        self._auto_conversion = False
        if var_form is None:
            # TODO after ansatz refactor num qubits can be set later so we do not have to have
            #      an operator to create a default
            if operator is not None:
                var_form = RY(operator.num_qubits)

        if optimizer is None:
            optimizer = SLSQP()

        # TODO after ansatz refactor we may still not be able to do this
        #      if num qubits is not set on var form
        if initial_point is None and var_form is not None:
            initial_point = var_form.preferred_init_points

        self._max_evals_grouped = max_evals_grouped

        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point,
                         quantum_instance=quantum_instance)
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback

        #Temporary fix until I get things cleaned up
        self._rbm_params = rbm_params

        # TODO if we ingest backend we can set expectation through the factory here.
        self._expectation_value = expectation_value
        self.operator = operator
        self.aux_operators = aux_operators

        self._eval_count = 0
        logger.info(self.print_settings())

        self._var_form_params = None
        if self.var_form is not None:
            self._var_form_params = ParameterVector('θ', self.var_form.num_parameters)

    @property
    def operator(self) -> Optional[OperatorBase]:
        """ Returns operator """
        return self._operator

    @operator.setter
    def operator(self, operator: OperatorBase) -> None:
        """ set operator """
        if isinstance(operator, LegacyBaseOperator):
            operator = operator.to_opflow()
        self._operator = operator
        self._check_operator_varform()
        if self._expectation_value is not None:
            self._expectation_value.operator = self._operator

    @property
    def expectation_value(self):
        """ Makes aux ops obsolete, as we can now just take
        the expectations of the ops directly. """
        return self._expectation_value

    @expectation_value.setter
    def expectation_value(self, exp):
        # TODO throw an error if operator is different from exp's operator?
        #  Or don't store it at all, only in exp?
        self._expectation_value = exp

    @property
    def aux_operators(self) -> List[LegacyBaseOperator]:
        """ Returns aux operators """
        return self._aux_operators

    @aux_operators.setter
    def aux_operators(self, aux_operators: List[LegacyBaseOperator]) -> None:
        """ Set aux operators """
        # This is all terrible code to deal with weight 0-qubit None aux_ops.
        self._aux_op_nones = None
        if isinstance(aux_operators, list):
            self._aux_op_nones = [op is None for op in aux_operators]
            zero_op = I.tensorpower(self.operator.num_qubits) * 0.0
            converted = [op.to_opflow() if op else zero_op for op in aux_operators]
            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes. TODO fix
            converted = [zero_op if op == 0 else op for op in converted]
            aux_operators = ListOp(converted)
        elif isinstance(aux_operators, LegacyBaseOperator):
            aux_operators = [aux_operators.to_opflow()]
        self._aux_operators = aux_operators
        if self.var_form is not None:
            self._var_form_params = ParameterVector('θ', self.var_form.num_parameters)
        self._parameterized_circuits = None

    @VQAlgorithm.var_form.setter
    def var_form(self, var_form: VariationalForm):
        """ Sets variational form """
        VQAlgorithm.var_form.fset(self, var_form)
        self._var_form_params = ParameterVector('θ', var_form.num_parameters)
        if self.initial_point is None:
            self.initial_point = var_form.preferred_init_points
        self._check_operator_varform()

    def _check_operator_varform(self):
        if self.operator is not None and self.var_form is not None:
            if self.operator.num_qubits != self.var_form.num_qubits:
                # TODO After Ansatz update we should be able to set in the
                #      number of qubits to var form. Important since use by
                #      application stack of VQE the user may be able to set
                #      a var form but not know num_qubits. Whether any smarter
                #      settings could be optionally done by VQE e.g adjust depth
                #      is TBD. Also this auto adjusting might not be reasonable for
                #      instance UCCSD where its parameterization is much closer to
                #      the specific problem and hence to the operator
                raise AquaError("Variational form num qubits does not match operator")

    @VQAlgorithm.optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        """ Sets optimizer """
        super().optimizer = optimizer
        if optimizer is not None:
            optimizer.set_max_evals_grouped(self._max_evals_grouped)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__)
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        if self._var_form is not None:
            ret += "{}".format(self._var_form.setting)
        else:
            ret += 'var_form has not been set'
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def _run(self) -> 'VQEResult':
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """

        # TODO delete instances of self._auto_conversion
        # TODO delete all instances of self._use_simulator_snapshot_mode
        # TODO make Expectations throw warnings more aggressively for
        #  incompatible operator primitives

        if self.operator is None:
            raise AquaError("Operator was never provided")

        self._operator = self.operator

        if self._expectation_value is None:
            self._expectation_value = ExpectationBase.factory(operator=self._operator,
                                                              backend=self._quantum_instance)
        self._aux_operators = self.aux_operators
        if self._auto_conversion:
            self._operator = \
                self._config_the_best_mode(self._operator, self._quantum_instance.backend)
            for i in range(len(self._aux_operators)):
                if self._aux_operators[i] is None:
                    continue
                if not self._aux_operators[i].is_empty():
                    self._aux_operators[i] = \
                        self._config_the_best_mode(self._aux_operators[i],
                                                   self._quantum_instance.backend)

        # sanity check
        if isinstance(self._operator, MatrixOperator) and not self._quantum_instance.is_statevector:
            raise AquaError("Non-statevector simulator can not work "
                            "with `MatrixOperator`, either turn ON "
                            "auto_conversion or use the proper "
                            "combination between operator and backend.")

        self._use_simulator_snapshot_mode = (
            is_aer_provider(self._quantum_instance.backend)
            and self._quantum_instance.run_config.shots == 1
            and not self._quantum_instance.noise_config
            and isinstance(self._operator,
                           (WeightedPauliOperator, TPBGroupedWeightedPauliOperator)))

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0

        if self._rbm_params is None:
            cost_fn = self._energy_evaluation
        else:
            cost_fn=self._DNN_energy_evaluation

        vqresult = self.find_minimum(initial_point=self.initial_point,
                                     var_form=self.var_form,
                                     cost_fn=cost_fn,
                                     optimizer=self.optimizer)

        # TODO remove all former dictionary logic
        self._ret = {}
        self._ret['num_optimizer_evals'] = vqresult.optimizer_evals
        self._ret['min_val'] = vqresult.optimal_value
        self._ret['opt_params'] = vqresult.optimal_point
        self._ret['eval_time'] = vqresult.optimizer_time

        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self.get_optimal_cost()])
        self._ret['eigvecs'] = np.asarray([self.get_optimal_vector()])

        result = VQEResult()
        result.combine(vqresult)
        result.eigenvalue = vqresult.optimal_value + 0j
        result.eigenstate = self.get_optimal_vector()

        if self.aux_operators:
            self._eval_aux_ops()
            # TODO remove when ._ret is deprecated
            result.aux_operator_eigenvalues = self._ret['aux_ops'][0]

        result.cost_function_evals = self._eval_count

        return result

    def _eval_aux_ops(self, threshold=1e-12):
        # Create a new ExpectationBase object to evaluate the auxops.
        expect = self.expectation_value.__class__(operator=self.aux_operators,
                                                  backend=self._quantum_instance,
                                                  state=CircuitStateFn(self.get_optimal_circuit()))
        values = np.real(expect.compute_expectation())
        # Discard values below threshold
        # TODO remove reshape when ._ret is deprecated
        aux_op_results = (values * (np.abs(values) > threshold))
        # Terribly hacky code to deal with weird legacy aux_op behavior
        self._ret['aux_ops'] = [None if is_none else [result]
                                for (is_none, result) in zip(self._aux_op_nones, aux_op_results)]
        self._ret['aux_ops'] = np.array([self._ret['aux_ops']])

    def compute_minimum_eigenvalue(
            self, operator: Optional[OperatorBase] = None,
            aux_operators: Optional[List[OperatorBase]] = None) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    # This is the objective function to be passed to the optimizer that is used for evaluation
    def _DNN_energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            Union(float, list[float]): energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        parameter_sets = np.split(parameters, num_parameter_sets)
        
        # TODO this is a hack to make AdaptVQE work, but it should fixed in adapt and deleted.
        if self._expectation_value is None:
            self._expectation_value = ExpectationBase.factory(operator=self._operator,
                                                              backend=self._quantum_instance)
            
        if not self._expectation_value.state:
            
            ansatz_circuit_op = CircuitStateFn(
                self._var_form.construct_circuit(self._var_form_params))
            self._expectation_value.state = ansatz_circuit_op



        param_bindings = {self._var_form_params: parameter_sets}
        
        #import ipdb; ipdb.set_trace()
        
        #### DNN Additions ####

        #Sample bases
        num_samples = self._rbm_params['n_samples_data']
        pauli_list = [str(op.primitive) for op in self._operator.oplist]  
        basis_str_list = [utils.SampleBasis(self._operator.num_qubits, 'ham', pauli_list) for basis in range(num_samples)]
        print(str(num_samples)+" Bases were sampled from the Hamiltonian's pauli string decomposition")
        
        #print(basis_str_list[:5],"...",basis_str_list[-5:])

        print("-")
        print('Generating sampling circuits for training bases...')
        #Get circuits to sample from these bases
        basis_op_list = ListOp([PauliOp(Pauli.from_label(basis)) for basis in basis_str_list])
        conv_op_list = PauliBasisChange(replacement_fn= lambda circuit_op, dest: circuit_op, traverse=True).convert(basis_op_list)
        rotated_trial_states = conv_op_list.compose(self._expectation_value.state).reduce()

        

        ham_bases = ([str(op.primitive) for op in self._operator.oplist])
        ham_coeffs = ([np.float(op.coeff) for op in self._operator.oplist])
      
        start_time = time()

        
        # Execute the basis sampling circuits. this method will apropriately batch the jobs for our backend.
        print("Evaluating circuits to build dataset:")

        cs = CircuitSampler.factory(backend=self._quantum_instance)
        cs.quantum_instance.run_config.shots = 100
        cs._snapshot = False
        result = cs.convert(rotated_trial_states, params=param_bindings)[0]
        basis_samples = [[float(bit) for bit in list([* res._primitive][0])] for res in result]


        
        #print(basis_samples[:5],"\n",basis_samples[-5:])


        (means, stds)= self._train_and_eval_rbm(num_qubits=self.operator.num_qubits, 
                                        samples=basis_samples, 
                                        bases=basis_str_list, 
                                        ham_bases=ham_bases, 
                                        ham_coeffs=ham_coeffs)

        ####---------------####
        print(means)
        #print(self._expectation_value)
        means_0 = np.real(self._expectation_value.compute_expectation(params=param_bindings))
        print(means_0)

        if self._callback is not None:
            stds = np.real(
                self._expectation_value.compute_standard_deviation(params=param_bindings))
            for i, param_set in enumerate(parameter_sets):
                self._eval_count += 1
                self._callback(self._eval_count, param_set, means[i], stds[i])
        # TODO I would like to change the callback to the following, to allow one to access an
        #  accurate picture of the evaluation steps, and to distinguish between single
        #  energy and gradient evaluations.
        if self._callback is not None and False:
            self._callback(self._eval_count, parameter_sets, means, stds)

        end_time = time()
        print((end_time - start_time) * 1000)
        print((end_time - start_time))
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s',
                    means, (end_time - start_time) * 1000, self._eval_count)

        return means if len(means) > 1 else means[0]


    def _train_and_eval_rbm(self, 
                            num_qubits,
                            samples=None,
                            bases=None,
                            path_to_sample=None,
                            path_to_bases=None,
                            psi=None,
                            alpha=1,
                            learning_rate=0.001,
                            n_samples=10000,
                            n_samples_data=1000,
                            n_epochs=10000,
                            ham_bases=None,
                            ham_coeffs=None):

        if ham_bases is None:
            ham_bases = []
        if ham_coeffs is None:
            ham_coeffs = []


        if self._rbm_params is not None:
            if 'alpha' in self._rbm_params:
                alpha = self._rbm_params['alpha']
            if 'learning_rate' in self._rbm_params:
                learning_rate = self._rbm_params['learning_rate']
            if 'n_samples' in self._rbm_params:
                n_samples = self._rbm_params['n_samples']
            if 'n_samples_data' in self._rbm_params:
                n_samples_data = self._rbm_params['n_samples_data']
            if 'n_epochs' in self._rbm_params:
                n_epochs = self._rbm_params['n_epochs']

        

        mpi_rank = nk.MPI.rank()

        # Read the total number of qubits
        N = num_qubits

        # Create a 1-dimensional lattice with open boundaries
        graph = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)

        # Create the Hilbert space
        hilbert = nk.hilbert.Qubit(graph=graph)
        assert (N == hilbert.size)

        hamiltonian = utils.GenerateHamiltonian(hilbert, ham_bases, ham_coeffs)
        # Run exact diagonalization using the Lanczos algorithm for the ground state
        # eigensystem = nk.exact.lanczos_ed(hamiltonian, first_n=1, compute_eigenvectors=True)
        hamiltonian_dense = hamiltonian.to_sparse().todense()
        (eigenvalues, eigenstates) = np.linalg.eigh(hamiltonian_dense)
        # Ground state wavefunction
        psi = np.reshape(np.asarray(eigenstates[:, 0]), [eigenstates.shape[0]])
        # Ground state energy
        E_0 = eigenvalues[0]

        rotations, tr_samples, tr_bases = utils.LoadData(hilbert, samples=samples, bases=bases)

        if (n_samples > tr_samples.shape[0]): n_samples = tr_samples.shape[0]
        training_samples, training_bases = utils.SliceData(tr_samples, tr_bases, n_samples)
        ma = nk.machine.RbmSpin(hilbert=hilbert, alpha=alpha)
        ma.init_random_parameters(seed=1234, sigma=0.01)

        # Sampler
        sa = nk.sampler.ExactSampler(machine=ma)
        # sa = nk.sampler.MetropolisLocal(machine=ma)
        # sa = nk.sampler.MetropolisLocalPt(machine=ma,n_replicas=16)

        # Optimizer
        # op = nk.optimizer.Sgd(learning_rate=learning_rate)
        # op = nk.optimizer.AdaDelta()
        op = nk.optimizer.RmsProp(learning_rate=learning_rate, beta=0.9, epscut=1.0e-6)

        qst = nk.Qsr(
            sampler=sa,
            optimizer=op,
            rotations=rotations,
            samples=training_samples[int(0.1 * len(training_samples)):],
            bases=training_bases[int(0.1 * len(training_samples)):],
            n_samples=n_samples,
            n_samples_data=n_samples_data,
            sr=nk.optimizer.SR(diag_shift=0.1))
        qst.add_observable(hamiltonian, "Energy")

        print("-\n")
        print("Training RBM...")
    
        iters = []
        overlap = []
        if psi is None: overlap.append("NA")
        nll = []
        E_nn = []
        E_var = []
        delta_E = []
        delta_E_rel = []
        for ep in qst.iter(n_epochs, 50):
            iters.append(ep)
            obs = qst.get_observable_stats()
            print(obs)
            if (mpi_rank == 0):
                # Compute fidelity with exact state
                psi_nn = ma.to_array()

                if psi is not None: overlap.append(utils.Overlap(psi_nn, psi))
                # Compute NLL on training data
                import ipdb;
                #ipdb.set_trace()
                nll.append(qst.nll(rotations=rotations,
                                   samples=training_samples[0:int(0.1 * len(training_samples))],
                                   bases=training_bases[0:int(0.1 * len(training_bases))],
                                   log_norm=ma.log_norm()))
                E_nn.append(obs["Energy"].mean.real)
                E_var.append(obs["Energy"].variance)
                delta_E.append(abs(obs["Energy"].mean.real - E_0))
                delta_E_rel.append(abs(obs["Energy"].mean.real - E_0) / abs(E_0))
                print('Ep = %d   ' % ep, end='')
                print('NLL = %.5f   ' % nll[-1], end='')
                print('Ov = %.5f   ' % overlap[-1], end='')
                print('E = %.5f   ' % E_nn[-1], end='')
                print('varE = %.2E   ' % E_var[-1], end='')
                print('d_rel = %.2E   ' % (delta_E_rel[-1]), end='')
                print('d_abs = %.2E   ' % delta_E[-1], end='')
                print()

        # TODO: return a more sensible and useful results object.
        return ([E_nn[-1]], [np.sqrt(E_var[-1])])
    
    # This is the objective function to be passed to the optimizer that is used for evaluation
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            Union(float, list[float]): energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        parameter_sets = np.split(parameters, num_parameter_sets)

        # TODO this is a hack to make AdaptVQE work, but it should fixed in adapt and deleted.
        if self._expectation_value is None:
            self._expectation_value = ExpectationBase.factory(operator=self._operator,
                                                              backend=self._quantum_instance)

        if not self._expectation_value.state:
            ansatz_circuit_op = CircuitStateFn(
                self._var_form.construct_circuit(self._var_form_params))
            self._expectation_value.state = ansatz_circuit_op
        param_bindings = {self._var_form_params: parameter_sets}

        start_time = time()
        means = np.real(self._expectation_value.compute_expectation(params=param_bindings))

        if self._callback is not None:
            stds = np.real(
                self._expectation_value.compute_standard_deviation(params=param_bindings))
            for i, param_set in enumerate(parameter_sets):
                self._eval_count += 1
                self._callback(self._eval_count, param_set, means[i], stds[i])
        # TODO I would like to change the callback to the following, to allow one to access an
        #  accurate picture of the evaluation steps, and to distinguish between single
        #  energy and gradient evaluations.
        if self._callback is not None and False:
            self._callback(self._eval_count, parameter_sets, means, stds)

        end_time = time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s',
                    means, (end_time - start_time) * 1000, self._eval_count)

        return means if len(means) > 1 else means[0]

    
    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']


class VQEResult(VQResult, MinimumEigensolverResult):
    """ VQE Result."""

    @property
    def cost_function_evals(self) -> int:
        """ Returns number of cost optimizer evaluations """
        return self.get('cost_function_evals')

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """ Sets number of cost function evaluations """
        self.data['cost_function_evals'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'eval_count':
            warnings.warn('eval_count deprecated, use cost_function_evals property.',
                          DeprecationWarning)
            return super().__getitem__('cost_function_evals')

        try:
            return VQResult.__getitem__(self, key)
        except KeyError:
            return MinimumEigensolverResult.__getitem__(self, key)
