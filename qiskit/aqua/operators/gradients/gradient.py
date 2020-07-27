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

"""The base interface for Aqua's gradients."""

from typing import Optional, Union, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.aqua.operators import OperatorBase

class Gradiemt(GradientBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields the first-order gradient with respect to the circuit parameters.
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
        operator: OperatorBase = None,
        params: Optional[List] = None,
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
        if method == 'param_shift':
            return self.parameter_shift(operator, params)
        if method == 'ancilla':
            return self.ancilla_hessian(params)
