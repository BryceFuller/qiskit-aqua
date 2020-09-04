
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
from functools import reduce
import numpy as np
from jax import grad, jit

from typing import Optional, Union, List, Tuple

from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.aqua.aqua_globals import AquaError
from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.list_ops import ListOp, ComposedOp, SummedOp, TensoredOp
from qiskit.aqua.operators import Zero, One, CircuitStateFn, StateFn

from .hessian_lin_comb import HessianLinComb
from .hessian_param_shift import HessianParamShift

from ..gradient_base import GradientBase
from ..gradient import Gradient


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
            method: The method used to compute the gradient. Either 'param_shift' or 'fin_diff' or 'lin_comb'.

        Returns:
            gradient_operator: An operator whose evaluation yeild the Hessian
        """
        # if input is a tuple instead of a list, wrap it into a list
        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        if isinstance(params, (ParameterVector, List)):
            # Case: a list of parameters were given, compute the Hessian for all param pairs
            if all(isinstance(param, Parameter) for param in params):
                return ListOp([ListOp([self.convert(operator, (p0, p1), method) for p1 in params]) for p0 in params])
            # Case: a list was given containing tuples of parameter pairs.
            # Compute the Hessian entries corresponding to these pairs of parameters. 
            elif all(isinstance(param, tuple) for param in params):
                return ListOp([self.convert(operator, param_pair, method) for param_pair in params])

        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self.factor_coeffs_out_of_composed_op(expec_op)
        return self.autograd(cleaned_op, params, method)

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
            # Case: a list of parameters were given, compute the Hessian for all param pairs
            if all(isinstance(param, Parameter) for param in params):
                return ListOp([ListOp([self.autograd(operator, (p0, p1), method) for p1 in params]) for p0 in params])
            # Case: a list was given containing tuples of parameter pairs.
            # Compute the Hessian entries corresponding to these pairs of parameters. 
            elif all(isinstance(param, tuple) for param in params):
                return ListOp([self.autograd(operator, param_pair, method) for param_pair in params])
    
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

            grad_op = 0
            #Avoid creating operators that will evaluate to zero
            if dd_op != ~Zero@One and not is_coeff_c(coeff, 0):
                grad_op += coeff*dd_op
            if d0_op != ~Zero@One and not is_coeff_c(d1_coeff, 0):
                grad_op += d1_coeff*d0_op
            if d1_op != ~Zero@One and not is_coeff_c(d0_coeff, 0):
                grad_op += d0_coeff*d1_op
            if not is_coeff_c(dd_coeff, 0):
                grad_op += dd_coeff*op

            if grad_op == 0:
                return ~Zero@One
                
            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):
            if not is_coeff_c(operator._coeff, 1.):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO if this is a sum of circuit state fns - traverse including autograd
            if isinstance(operator[-1], (CircuitStateFn)):
                pass
            else:
                raise TypeError('The gradient framework is compatible with states that are given as CircuitStateFn')

            if method == 'param_shift':
                return HessianParamShift().convert(operator, params)
            elif method == 'fin_diff':
                return HessianParamShift().convert(operator, params, analytic=False)
            elif method == 'lin_comb':
                return HessianLinComb().convert(operator, params)

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
             
            if operator.grad_combo_fn:
                grad_combo_fn = operator.grad_combo_fn
            else:

                try:
                    grad_combo_fn = jit(grad(operator._combo_fn, holomorphic=True))
                except Exception:
                    raise TypeError('This automatic differentiation function is based on JAX. Please use import '
                              'jax.numpy as jnp instead of import numpy as np when defining a combo_fn.')

            # f(g_1(x), g_2(x)) --> df/dx = df/dg_1 dg_1/dx + df/dg_2 dg_2/dx
            return ListOp([ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                          combo_fn=lambda x: np.dot(x[0], x[1]))

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
