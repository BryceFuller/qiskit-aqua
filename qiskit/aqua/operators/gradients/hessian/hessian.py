
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

"""The module to compute Hessians."""

from typing import Optional, Union, List, Tuple

from qiskit.circuit import Parameter
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.list_ops.list_op import ListOp

from ..gradient_base import GradientBase


class Hessian(GradientBase):
    """Compute the Hessian of a expected value."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Tuple[Parameter, Parameter],
                                       List[Tuple[Parameter, Parameter]]]] = None,
                method: str = 'param_shift') -> OperatorBase:
        """
        Args:
            operator: The measurement operator we are taking the gradient of
            operator:  The operator corresponding to our state preparation circuit
            params: The parameters we are taking the gradient with respect to
            method: The method used to compute the gradient. Either 'param_shift' or 'ancilla'.

        Returns:
            gradient_operator: An operator whose evaluation yeild the Hessian
        """
        # if input is a tuple instead of a list, wrap it into a list
        if isinstance(params, tuple):
            is_tuple = True
            params = [params]
        else:
            is_tuple = False

        if method == 'param_shift':
            hessian = ListOp(
                [self.parameter_shift(self.parameter_shift(operator, pair[0]), pair[1])
                 for pair in params]
            )
        elif method == 'ancilla':
            hessian = self.ancilla_hessian(params)

        if is_tuple:  # if input was not a list extract the single operator from the list op
            return hessian.oplist[0]
        return hessian

    def ancilla_hessian(self, params):
        """TODO"""
        raise NotImplementedError  # TODO
