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
from typing import Optional, Callable, Union, List, Tuple
import logging
from functools import partial, reduce
import numpy as np
from copy import deepcopy


from ...list_ops.list_op import ListOp
from ...list_ops.summed_op import SummedOp
from ...list_ops.composed_op import ComposedOp


from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.aqua.operators import OperatorBase
from ..gradient_base import GradientBase


class GradientParamShift(GradientBase):
    """Compute the gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω with the parameter shift method."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None,
                analytic: bool = True
                ) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to our quantum state we are taking the
                gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
            analytic: If True use the parameter shift rule to compute analytic gradients,
                      else use a finite difference approach

        Returns:
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        # TODO Look through state and decompose gates which cannot be evaluated with the parameter
        # shift rule. This seems like it could be it's own converter??
        # decomposed_state = self.decompose_to_two_unique_eigenval(state_operator, params)

        # TODO Note to above --> all are pi/2
        # parameter_shift will return a ListOp of the same size as params
        # one SummedOp per parameter.
        if analytic:
            return self.parameter_shift(operator, params)
            #TODO move the logic for parameter shifting from gradient_base here
        else:
            pass
            # TODO add finite difference

    def parameter_shift(self,
                        operator: OperatorBase,
                        params: Union[Parameter, ParameterVector, List]) -> OperatorBase:

        r"""
        Args:
            operator: the operator containing circuits we are taking the derivative of
            params: The parameters (ω) we are taking the derivative with respect to. If
                    a ParameterVector is provided, each parameter will be shifted.
        Returns:
            param_shifted_op: A ListOp of SummedOps corresponding to [r*(V(ω_i + π/2) - V(ω_i - π/2)) for w_i in params]
        """
        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.parameter_shift(operator, param) for param in params]
            absent_params = [params[i] for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError("The following parameters do not appear in the provided operator: ", absent_params)
            return ListOp(absent_params)

        # by this point, it's only one parameter
        param = params

        if not isinstance(param, Parameter):
            raise ValueError
        if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
            return_op = operator.traverse(partial(self.parameter_shift, params=param))

            # Remove any branch of the tree where the relevant parameter does not occur
            trimmed_oplist = [op for op in return_op.oplist if op is not None]
            # If all branches are None, remove the parent too
            if len(trimmed_oplist) == 0:
                return None
            # Rebuild the operator with the trimmed down oplist
            properties = {'coeff': return_op._coeff, 'abelian': return_op._abelian}
            if return_op.__class__ == ListOp:
                properties['combo_fn'] = return_op.combo_fn
            return return_op.__class__(oplist=trimmed_oplist, **properties)

        else:

            circs = self.get_unique_circuits(operator)

            if len(circs) > 1:
                # Understand how this happens
                raise Error
            elif len(circs) == 0:
                print("No circuits found in: ", operator)
                return operator
            circ = circs[0]

            if param not in circ._parameter_table:
                return None

            shifted_ops = []
            for i in range(len(circ._parameter_table[param])):
                # Implement the gradient for this particular rotation.
                # Here is where future logic will go that decomposes more
                # complicated rotations
                pshift_op = deepcopy(operator)
                mshift_op = deepcopy(operator)

                # We need the circuit objects of the newly instantiated operators
                pshift_circ = self.get_unique_circuits(pshift_op)[0]
                mshift_circ = self.get_unique_circuits(mshift_op)[0]

                pshift_gate = pshift_circ._parameter_table[param][i][0]
                mshift_gate = mshift_circ._parameter_table[param][i][0]

                # TODO here chain rule parameter shift

                assert len(pshift_gate.params) == 1, "Circuit was not properly decomposed"

                #The parameter could be a parameter expression
                p_param = pshift_gate.params[0]
                m_param = mshift_gate.params[0]

                # TODO: Add check that asserts a NotImplementedError if gate is not a standard qiskit gate.
                # Assumes the gate is a pauli rotation!
                shift_constant = 0.5

                pshift_gate.params[0] = (p_param + (np.pi / (4*shift_constant)))
                mshift_gate.params[0] = (m_param - (np.pi / (4*shift_constant)))

                shifted_op = shift_constant * (pshift_op - mshift_op)

                # If the rotation angle is actually a parameter expression of param, then handle the chain rule
                if pshift_gate.params[0] != param and isinstance(pshift_gate.params[0], ParameterExpression):
                        expr_grad = self.parameter_expression_grad(pshift_gate.params[0], param)
                        shifted_op *= expr_grad

                shifted_ops.append(shifted_op)

            return shifted_op.reduce()
            #return SummedOp(shifted_ops).reduce()

