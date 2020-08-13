
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

""" Hessian Class """

from typing import Optional, Callable, Union, List, Tuple
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.primitive_ops.circuit_op import CircuitOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.operator_globals import H, S, I
from ..gradient_bas import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

logger = logging.getLogger(__name__)


class Hessian(GradientBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields the second-order gradient with respect to the circuit parameters.
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
        operator: OperatorBase = None,
        param_pairs: Union[Tuple, List] = None,
        method: str = 'param_shift') -> OperatorBase:
        r"""
        Args:
            operator: The measurement operator we are taking the gradient of
            state_operator:  The operator corresponding to our state preparation circuit
            parameters: The parameters we are taking the gradient with respect to
            method: The method used to compute the gradient. Either 'param_shift' or 'ancilla'
        Returns:
            gradient_operator: An operator whose evaluation yeild the Hessian
        """
        if isinstance(param_pairs, Tuple):
            if method == 'param_shift':
                return self.parameter_shift_grad(self.parameter_shift(operator, param_pairs[0]), param_pairs[1])
            if method == 'ancilla':
                return self.ancilla_hessian(param_pairs)

        # Todo fix the ancilla method here
        return ListOp([self.parameter_shift(self.parameter_shift(operator, pair[0]), pair[1]) for pair in param_pairs])

    # Todo add ancilla hessian here
