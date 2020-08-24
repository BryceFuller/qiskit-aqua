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

"""The operator gradient module."""

from typing import Optional, Union, List
import logging
from functools import partial

from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from ..gradient_base import GradientBase

logger = logging.getLogger(__name__)


class OperatorGradient(GradientBase):
    """Computing the operator gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dθ."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
                ) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to our quantum state we are taking the
                gradient of: |ψ(ω).
            params: The parameters we are taking the gradient with respect to: θ

        Returns:
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        def contains_param(operator, param):
            if isinstance(operator.coeff, (Parameter, ParameterExpression)):
                if param in operator.coeff.parameters:
                    return True
            return False

        def prune_meas_op(meas_op, param):
            if isinstance(meas_op, StateFn):
                if meas_op.is_measurement:
                    new_op = prune_meas_op(meas_op.primitive, param)
                    if new_op is not None:
                        return ~StateFn(new_op, coeff=(meas_op.coeff))
                    return

            if isinstance(meas_op, ListOp):
                if isinstance(meas_op, ComposedOp):
                    raise NotImplementedError
                traversed = meas_op.traverse(partial(prune_meas_op, param=param))
                oplist = [op for op in traversed.oplist if op is not None]
                if len(oplist) == 0:
                    return
                return type(meas_op)(oplist, coeff=meas_op.coeff)
            else:
                if contains_param(meas_op, param):
                    return meas_op / param
            return

        if isinstance(params, (ParameterVector, List)):
            return ListOp([self.convert(operator, param) for param in params])
        else:
            param = params

        if contains_param(operator, param):
            return operator / param

        if isinstance(operator, StateFn):
            if operator.is_measurement:
                return prune_meas_op(operator, param)
            return operator
        elif isinstance(operator, ComposedOp):
            conv_ops = [self.convert(op, param) for op in operator]
            if (conv_ops)[0] is not None:
                return ComposedOp(conv_ops, coeff=operator.coeff)
        elif isinstance(operator, ListOp):
            pruned_list = [op for op in [self.convert(op, param)
                                         for op in operator] if op is not None]
            if len(pruned_list) == 0:
                return
            return type(operator)(pruned_list, coeff=operator.coeff)

        return
