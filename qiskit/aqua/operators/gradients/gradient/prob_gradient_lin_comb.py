# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" ProbabilityGradient Class """

from typing import Optional, Callable, Union, List, Dict
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import Aer, QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.primitive_ops.circuit_op import CircuitOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z
from qiskit.aqua.operators.converters.circuit_sampler import CircuitSampler
from qiskit.aqua.operators.gradients.gradient import StateGradientAncilla

logger = logging.getLogger(__name__)


class ProbabilityGradientAncilla(StateGradientAncilla):
    r"""
    Special Case of the StateGradient where the gradient_operator is a projector on all possible basis states.
    This computes the gradients of the sampling probabilities of the basis states rather than an expectation value.

    We are interested in computing:
    d⟨ψ(ω)|ψ(ω)〉/ dω for ω in params
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None,
                param_bindings: Dict = None,
                backend: Union[QuantumInstance, BaseBackend] = None) -> OperatorBase:
        r"""
        Args
            state_operator: |ψ(ω)〉, The operator corresponding to our quantum state we are taking the gradient of ()
            params: The parameters we are taking the gradient with respect to
        """

        if operator.is_measurement:
            raise AquaError('Probability gradients are computed with respect to states instead of expectation values.'
                            'Please remove the measurement operator.')

        operator = super().convert(operator, params)

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        grad_gates = {}
        if isinstance(operator, CircuitStateFn):
            state_qc = operator.primitive
        elif isinstance(operator, DictStateFn) or isinstance(operator, VectorStateFn):
            state_qc = DictToCircuitSum.convert(operator).primitive
        else:
            raise TypeError('Ancilla gradients only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        for param, elements in state_qc._parameter_table.items():
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    grad_coeffs[param].append(c_g[0])
                    grad_gates[param].append(c_g[1])

        states = self._grad_states(gates_to_parameters, grad_coeffs, grad_gates)
        grad_op = I ^ state_qc.num_qubits ^ Z

        # TODO fix eval

        # evaluation_qubit = find_regs_by_name(qc, 'ancilla')
        # qregs_list = state_qc.qregs
        # index_evaluation_qubit = qregs_list.index(evaluation_qubit) # = state_qc.num_qubits + 1
        # op = op.to_matrix()
        # # Construct circuits to evaluate the expectation values
        # sampler = CircuitSampler(self._quantum_instance).convert(StateFn(circuit_item))
        # result = sampler.to_density_matrix()
        # prob_grad = partial_trace(op.dot(result), [index_evaluation_qubit])
        # results.append(list(np.diag(prob_grad.data)))

        return grad_op @ states
