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

from typing import Optional, Union, List

from qiskit.circuit import Parameter, ParameterVector
from qiskit.aqua.operators import OperatorBase
from qiskit.aqua.operators.gradients import GradientBase

from .state_gradient_lin_comb import StateGradientLinComb
from .state_gradient_param_shift import StateGradientParamShift


class StateGradient(GradientBase):
    r"""Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List]] = None,
                method: str = 'param_shift') -> OperatorBase:
        """
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient, can be either of
                ``'param_shift'`` or ``'lin_comb'``.

        Returns:
            An operator whose evaluation yields the Gradient

        Raises:
            NotImplementedError: If an unsupported ``mode`` is selected.
        """
        if method == 'param_shift':
            return StateGradientParamShift().convert(operator=operator, params=params)
        elif method == 'lin_comb':
            return StateGradientLinComb().convert(operator=operator, params=params)
        else:
            raise NotImplementedError('The chosen gradient method is not implemented. '
                                      'Please choose either ``param_shift`` or ``lin_comb``.')
