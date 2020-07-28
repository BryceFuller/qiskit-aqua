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

""" ProbabilityHessian Class """

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
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.primitive_ops.circuit_op import CircuitOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.operator_globals import H, S, I
from qiskit.aqua.operators.converters.circuit_sampler import CircuitSampler
from .state_hessian_param_shift import StateHessianParamShift

logger = logging.getLogger(__name__)


class ProbabilityHessianParamShift(StateHessianParamShift):
    r"""
    Special Case of the StateHessian where the hessian_operator is the identity
    (Actually there are some more nuanced differences on how this is computed in practice!)

    We are interested in computing:
    d⟨ψ(ω)|ψ(ω)〉/ dω for ω in params
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None,
                analytic: bool = True) -> OperatorBase:
        r"""
        Args
            state_operator: |ψ(ω)〉, The operator corresponding to our quantum state we are taking the hessian of ()
            params: The parameters we are taking the hessian with respect to
            analytic: If true compute an analytic hessian, else compute a finite difference approximations
        """
        # TODO add finite differences
        # TODO backend and param bindings?
        if operator.is_measurement:
            raise AquaError('Probability hessians are computed with respect to states instead of expectation values.'
                            'Please remove the measurement operator.')

        operator = super().convert(operator, params)

        # if backend is None:
        #     backend = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
        #                               shots=1024)
        #
        # # TODO: CHECK if state_operator contains an operator measurement, throw an error if it does!
        # sampler = CircuitSampler(backend=backend)
        #
        # # The CircuitStateFn's need to be wrapped so that the coefficient
        # # is stored elsewhere during circuit sampling
        # wrapped_op = self.wrap_circuit_statefn(operator)
        # sampled_op = sampler.convert(wrapped_op.bind_parameters(param_bindings))
        # grad_op = self.ravel(sampled_op).reduce()

        if isinstance(grad_op, ListOp):
            hessians = [grad.primitive for grad in grad_op.reduce()]
        else:
            hessians = [grad_op.primitive]
        return hessians

    def wrap_circuit_statefn(self, operator):
        if isinstance(operator, StateFn):
            return ListOp([operator / operator.coeff], coeff=operator.coeff)
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self.wrap_circuit_statefn))
        else:
            return operator

    def ravel(self, operator):
        if isinstance(operator, ListOp):
            if len(operator.oplist) == 1:
                return operator.oplist[0] * operator.coeff
            else:
                return operator.traverse(partial(self.ravel))
        else:
            return operator
