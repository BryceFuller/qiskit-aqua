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

""" OperatorGradient Class """

from typing import Optional, Callable, Union, List
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.primitive_ops.circuit_op import CircuitOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.list_ops.summed_op import SummedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.operator_globals import H, S, I
from ..gradient_base import GradientBase

logger = logging.getLogger(__name__)


class ObservableGradient(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for θ in params
    """

    def convert(self,
                state_operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            state_operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            target_operator: The measurement operator we are taking the gradient of: O(θ)
            params: The parameters we are taking the gradient with respect to: θ
        Returns
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        def contains_param(operator, param):
            if isinstance(operator.coeff, (Parameter, ParameterExpression)):
                if param in operator.coeff.parameters:
                    return True
            return False

        def prune_meas_op(meas_op, param):
            if isinstance(meas_op, StateFn):
                if meas_op.is_measurement == True:
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
                    return (meas_op / param)
            return

        if isinstance(params, (ParameterVector, List)):
            return ListOp([self.convert(state_operator, param) for param in params])
        else:
            param = params

        if contains_param(state_operator, param):
            return (state_operator / param)

        if isinstance(state_operator, StateFn):
            if state_operator.is_measurement:
                return prune_meas_op(state_operator, param)
            else:
                return state_operator


        elif isinstance(state_operator, ComposedOp):
            conv_ops = [self.convert(op, param) for op in state_operator]
            if (conv_ops)[0] is not None:
                return ComposedOp(conv_ops, coeff=state_operator.coeff)

        elif isinstance(state_operator, ListOp):
            pruned_list = [op for op in [self.convert(op, param) for op in state_operator] if op is not None]
            if len(pruned_list) == 0:
                return
            return type(state_operator)(pruned_list, coeff=state_operator.coeff)

        return
