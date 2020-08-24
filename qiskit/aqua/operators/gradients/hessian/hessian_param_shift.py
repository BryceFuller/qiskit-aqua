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

"""The module to compute the Hessian using the parameter shift method."""

from typing import Optional, Union, List, Tuple

from qiskit.aqua.operators import OperatorBase, ListOp
from qiskit.circuit import Parameter

from ..gradient_base import GradientBase


class HessianParamShift(GradientBase):
    """TODO"""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Tuple[Parameter, Parameter],
                                       List[Tuple[Parameter, Parameter]]]] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The measurement operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to

        Returns:
            An operator whose evaluation yeild the Hessian
        """
        if isinstance(params, tuple):
            return self.parameter_shift(self.parameter_shift(operator, params[0]), params[1])

        return ListOp(
            [self.parameter_shift(self.parameter_shift(operator, pair[0]), pair[1])
             for pair in params]
            )
