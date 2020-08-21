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

"""The module to compute the state gradient with the parameter shift rule."""

from typing import Optional, Union, List

from qiskit.circuit import Parameter, ParameterVector
from qiskit.aqua.operators import OperatorBase
from ..gradient_base import GradientBase


class StateGradientParamShift(GradientBase):
    """Compute the gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω with the parameter shift method."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
                ) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to our quantum state we are taking the
                gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """
        # TODO add finite difference
        # analytic: bool = True
        # analytic: If True compute an analytic gradient, else compute a finite difference
        #     approximations

        # TODO Look through state and decompose gates which cannot be evaluated with the parameter
        # shift rule. This seems like it could be it's own converter??
        # decomposed_state = self.decompose_to_two_unique_eigenval(state_operator, params)

        # TODO Note to above --> all are pi/2
        # parameter_shift will return a ListOp of the same size as params
        # one SummedOp per parameter.
        return self.parameter_shift(operator, params)
