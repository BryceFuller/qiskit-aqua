
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

    def autograd(self,
                operator: OperatorBase,
                params: Optional[Union[Tuple[Parameter, Parameter],
                                       List[Tuple[Parameter, Parameter]]]] = None,
                method: str = 'param_shift') -> OperatorBase: 

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        if isinstance(params, (ParameterVector, List)):
            return ListOp([ListOp([self.autograd(operator, (p0, p1), method) for p1 in params]) for p0 in params])
            
        # If a gradient is requested w.r.t a single parameter, then call the
        # Gradient() class' autograd method.   
        if isinstance(params, Parameter):
            return Gradient().autograd(operator, params, method)

        assert isinstance(params, Tuple) and len(params) == 2, "Parameters supplied in unsupported format"

        # by this point, it's only one parameter
        p0 = params[0]
        p1 = params[1]

        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0):
            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator/coeff
            # Get derivative of the operator (recursively)
            d0_op = self.autograd(op, p0, method)
            d1_op = self.autograd(op, p1, method)
            # ..get derivative of the coeff
            d0_coeff = self.parameter_expression_grad(coeff, p0)
            d1_coeff = self.parameter_expression_grad(coeff, p1)

            dd_op = self.autograd(op, params, method)
            dd_coeff = self.parameter_expression_grad(d0_coeff, p1)

            grad_op = None
            summands = np.array([0,0,0,0])

            #Avoid creating operators that will evaluate to zero
            if dd_op is not None and not is_coeff_c(coeff, 0):
                summands[0] = coeff*dd_op
            if d0_op is not None and not is_coeff_c(d1_coeff, 0):
                summands[0] = d1_coeff*d0_op
            if d1_op is not None and not is_coeff_c(d0_coeff, 0):
                summands[0] = d0_coeff*d1_op
            if not is_coeff_c(dd_coeff, 0):
                summands[0] = dd_coeff*op

            if np.all(summands == 0):
                return ~Zero@One

            grad_op = reduce(lambda x,y: x if y == 0 else x+y, summands)

            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):
            if not is_coeff_one(operator._coeff):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO if this is a sum of circuit state fns - traverse including autograd
            if isinstance(operator[-1], (CircuitStateFn, CircuitOp)):
                # TODO check if CircuitOp/ CircuitStateFn
                pass
                # Do some checks and decide how you're planning on taking the gradient.
                # for now we do param shift

            elif isinstance(operator[-1], (VectorStateFn, DictStateFn)):
                operator[-1] = DictToCircuitSum().convert(operator[-1])
                # Do LCU logic # TODO what's that?
            else:
                raise TypeError('Gradients only support operators whose states are either '
                                'CircuitStateFn, DictStateFn, or VectorStateFn.')

            if method == 'param_shift':
                return HessianParamShift().convert(operator, params)
            elif method == 'fin_diff':
                return HessianParamShift().convert(operator, params, analytic=False)
                # return self.parameter_shift(operator, param)
            elif method == 'lin_comb':
                return HessiantLinComb().convert(operator, params)
                # @CHRISTA, here is where you'd check if you need to
                # decompose some operator into circuits or do
                # something other than the parameter shift rule. # TODO is this what I need?

        # This is the recursive case where the chain rule is handled
        elif isinstance(operator, ListOp):
            grad_ops = [self.autograd(op, params, method) for op in operator.oplist]

            # Note that this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn.
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and fail.
            # An alternative is to check the byte code of the operator's combo_fn against the
            # default one.
            # This will work but look very ugly and may have other downsides I'm not aware of
            if operator._combo_fn == ListOp([])._combo_fn:
                print("default combo fn")
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
                print("SummedOp combo fn")
                return SummedOp(oplist=grad_ops)
            elif isinstance(operator, TensoredOp):
                return TensoredOp(oplist=grad_ops)
            # else:
            #     raise NotImplementedError
            #     # TODO!

            # NOTE! This will totally break if you try to pass a DictStateFn through a combo_fn
            # (for example, using probability gradients)
            # I think this is a problem more generally, not just in this subroutine.
            grad_combo_fn = self.get_grad_combo_fn(operator)
            return ListOp(oplist=operator.oplist+grad_ops, combo_fn=grad_combo_fn) # TODO why operator.oplist? -> only grad_ops

        elif isinstance(operator, StateFn):
            if operator._is_measurement:
                raise Exception  # TODO raise proper error!
                # Doesn't make sense to have a StateFn measurement
                # at the end of the tree.
        else:
            print(type(operator))
            print(operator)
            # TODO raise proper error!
            raise Exception("Control Flow should never have reached this point")
            # This should never happen. The base case in our traversal is reaching a ComposedOp or
            # a StateFn. If a leaf of the computational graph is not one of these two, then the
            # original operator we are trying to differentiate is not an expectation value or a
            # state. Thus it's not clear what the user wants.
