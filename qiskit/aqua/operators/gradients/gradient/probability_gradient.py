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

"""The base interface the probability gradients."""

from typing import Optional, Union

from qiskit.circuit import Parameter, ParameterVector
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.aqua.operators import OperatorBase


class ProbabilityGradient(GradientBase):
    """TODO"""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, Parameter]] = None,
                method: str = 'param_shift') -> OperatorBase:
        """
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient. Can be either
                of ``'param_shift'`` or ``'lin_comb'``.

        Returns:
            An operator whose evaluation yields the Gradient
        """
        pass
