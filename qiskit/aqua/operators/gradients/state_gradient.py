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

""" StateGradient Class """

from typing import Optional, Callable, Union, List
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..primitive_ops.pauli_op import PauliOp
from ..primitive_ops.circuit_op import CircuitOp
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..state_fns.state_fn import StateFn
from ..operator_globals import H, S, I
from ..expectations import PauliExpectation
from .gradient_base import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

logger = logging.getLogger(__name__)


class StateGradient(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for ω in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None,
                method: str = 'param_shift') -> OperatorBase:
        r"""
        Args
            tate_operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            target_operator: The measurement operator we are taking the gradient of: O(θ)
            params: The parameters we are taking the gradient wrt: ω
            method: The method used to compute the gradient. Either 'param_shift' or 'ancilla'
        Returns
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        # TODO Look through state and decompose gates which cannot be evaluated with the parameter shift rule
        # This seems like it could be it's own converter??
        # decomposed_state = self.decompose_to_two_unique_eigenval(state_operator, params)

        if method == 'param_shift':
            # parameter_shift will return a ListOp of the same size as params
            # one SummedOp per parameter.
            return self.parameter_shift(operator, params)
        if method == 'ancilla':
            pass
            #todo fix

