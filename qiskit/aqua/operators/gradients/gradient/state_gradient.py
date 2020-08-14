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

"""The base interface for Aqua's gradient."""

from typing import Optional, Union, Tuple, List
import sympy as sy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector, Instruction
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.aqua.operators.gradients.gradient.state_gradient_lin_comb import StateGradientLinComb
from qiskit.aqua.operators.gradients.gradient.state_gradient_param_shift import StateGradientParamShift
from qiskit.aqua.operators import OperatorBase, ListOp

class StateGradient(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for θ in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None,
                method: str = 'param_shift') -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            parameters: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient. ['param_shift', 'lin_comb']
                    Deprecated for observable gradient
        Returns:
            gradient_operator: An operator whose evaluation yields the Gradient
        """
        if method == 'param_shift':
            return StateGradientParamShift().convert(operator=operator, params=params)
        elif method == 'lin_comb':
            return StateGradientLinComb().convert(operator=operator, params=params)
        else:
            raise AquaError('The chosen gradient method is not implemented. Please choose either param_shift '
                            'or lin_comb.')
