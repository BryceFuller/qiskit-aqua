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
from qiskit.aqua import QuantumInstance
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..primitive_ops.pauli_op import PauliOp
from ..primitive_ops.circuit_op import CircuitOp
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..state_fns.state_fn import StateFn
from ..operator_globals import H, S, I
from ..converters.circuit_sampler import CircuitSampler
from .state_gradient import StateGradient

logger = logging.getLogger(__name__)


class ProbabilityGradient(StateGradient):
    r"""
    Special Case of the StateGradient where the gradient_operator is the identity
    (Actually there are some more nuanced differences on how this is computed in practice!)

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

        operator = super().convert(operator, params)

        if backend is None:
            backend = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                      shots=1024)

        # TODO: CHECK if state_operator contains an operator measurement, throw an error if it does!
        sampler = CircuitSampler(backend=backend)

        # The CircuitStateFn's need to be wrapped so that the coefficient
        # is stored elsewhere during circuit sampling
        wrapped_op = self.wrap_circuit_statefn(operator)
        sampled_op = sampler.convert(wrapped_op.bind_parameters(param_bindings))
        grad_op = self.ravel(sampled_op).reduce()

        if isinstance(grad_op, ListOp):
            gradients = [grad.primitive for grad in grad_op.reduce()]
        else:
            gradients = [grad_op.primitive]
        return gradients

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