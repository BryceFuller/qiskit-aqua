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

"""The base interface for Aqua's gradient."""
from collections.abc import Iterable
from functools import partial
from typing import Optional, Union, List, Callable
import numpy as np
from jax import grad, jit

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector, Instruction
from qiskit.aqua import AquaError

from qiskit.aqua.operators import (
    OperatorBase, ListOp, SummedOp, ComposedOp, TensoredOp, CircuitOp, CircuitStateFn, StateFn, DictStateFn,
    VectorStateFn, Zero, One, DictToCircuitSum, PauliExpectation
)

from .gradient_lin_comb import GradientLinComb
from .gradient_param_shift import GradientParamShift

from ..gradient_base import GradientBase
from ...operator_base import OperatorBase
from ...primitive_ops.primitive_op import PrimitiveOp
from ...primitive_ops.pauli_op import PauliOp
from ...primitive_ops.circuit_op import CircuitOp
from ...list_ops.list_op import ListOp
from ...list_ops.composed_op import ComposedOp
from ...list_ops.summed_op import SummedOp
from ...list_ops.tensored_op import TensoredOp
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...state_fns.vector_state_fn import VectorStateFn
from ...state_fns.dict_state_fn import DictStateFn
from ...operator_globals import H, S, I, Zero, One
from ...converters.converter_base import ConverterBase


class Gradient(GradientBase):
    """Convert an operator expression to the first-order gradient."""

    # TODO the arguments shouldn't differ, all additional parameters should go to the initializer
    # pylint: disable=arguments-differ
    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, Parameter, List[Parameter]]] = None,
                method: str = 'lin_comb'
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to.
            method: The method used to compute the state/probability gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``. Deprecated for observable gradient.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        # grad_combo_fn: Gradient for a custom operator combo_fn. The gradient for a standard
        #     ``ListOp`` or SymPy combo_fn is automatically computed.

        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        # TODO where is the param = None case handled?
        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.convert(operator, param, method) for param in params]
            # If autograd returns None, then the corresponding parameter was probably not present
            # in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params
                )
            return ListOp(param_grads)

        param = params

        #Preprocessing
        #TODO think of better names...
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self.factor_coeffs_out_of_composed_op(expec_op)
        return self.autograd(cleaned_op, param, method)

    def autograd(self,
                 operator: OperatorBase,
                 params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]],
                 method: str = 'param_shift'
                 ) -> OperatorBase:
        """TODO

        Args:
            operator: TODO
            params: TODO
            method: TODO

        Returns:
            TODO

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            AquaError: If the coefficent of the operator could not be reduced to 1.
            NotImplementedError: If operator is a TensoredOp  # TODO support this
            Exception: Unintended code is reached  # TODO proper warnings and errors
        """

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        if isinstance(params, (ParameterVector, List)):
            param_grads = [self.autograd(operator, param, method) for param in params]
            # If autograd returns None, then the corresponding parameter was probably not present
            # in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    'The following parameters do not appear in the provided operator: ',
                    absent_params
                )
            return ListOp(param_grads)

        # by this point, it's only one parameter
        param = params
        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0):

            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator / coeff
            # Get derivative of the operator (recursively)
            d_op = self.autograd(op, param, method)
            # ..get derivative of the coeff
            d_coeff = self.parameter_expression_grad(coeff, param)
            

            grad_op = 0
            if d_op != ~Zero@One and not is_coeff_c(coeff, 0.0):
                grad_op += coeff*d_op
            if op != ~Zero@One and not is_coeff_c(d_coeff, 0.0):
                grad_op += d_coeff*op 
            if grad_op == 0:
                grad_op = ~Zero@One
            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):
            # Gradient of an expectation value
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO if this is a sum of circuit state fns - traverse including autograd
            if isinstance(operator[-1], (CircuitStateFn)):
                pass
                # Do some checks and decide how you're planning on taking the gradient.
                # for now we do param shift
            else:
                raise TypeError('The gradient framework is compatible with states that are given as CircuitStateFn')

            if method == 'param_shift':
                return GradientParamShift().convert(operator, param)
            elif method == 'fin_diff':
                return GradientParamShift().convert(operator, param, analytic=False)
            elif method == 'lin_comb':
                return GradientLinComb().convert(operator, param)

        elif isinstance(operator, CircuitStateFn):
            # Gradient of an a state's sampling probabilities
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            if method == 'param_shift':
                return GradientParamShift().convert(operator, param)
            elif method == 'fin_diff':
                return GradientParamShift().convert(operator, param, analytic=False)
            elif method == 'lin_comb':
                return GradientLinComb().convert(operator, param)

        # Handle the chain rule
        elif isinstance(operator, ListOp):
            grad_ops = [self.autograd(op, param, method) for op in operator.oplist]

            # Note: this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn.
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and raise an error.
            # An alternative is to check the byte code of the operator's combo_fn against the
            # default one.
            # This will work but look very ugly and may have other downsides I'm not aware of
            if operator._combo_fn == ListOp([])._combo_fn:
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
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

         

    # ---------------------------------------------------------------------

    # TODO I commented this
    # ---------------------------------------------------------------------
#def get_grad_combo_fn(self, operator: ListOp) -> Callable:
    #    """Get the derivative of the operator combo_fn.
    #
    #    Args:
    #        operator: The operator for whose combo_fn we want to get the gradient.
    #
    #    Returns:
    #        function which evaluates the partial gradient of operator._combo_fn
    #        with respect to each element of operator.oplist
    #
    #    Raises:
    #        Exception: If the operator is a ``ComposedOp``.
    #        Exception: If the gradient combo function could be differentiated.
    #    """
    # 
    #       return ListOp(oplist=operator.oplist+grad_ops, combo_fn=grad_combo_fn)
    #
    #     elif isinstance(operator, StateFn):
    #         if operator._is_measurement:
    #             raise Exception  # TODO raise proper error!
    #             # Doesn't make sense to have a StateFn measurement
    #             # at the end of the tree.
    #     else:
    #         print(type(operator))
    #         print(operator)
    #         # TODO raise proper error!
    #         raise Exception("Control Flow should never have reached this point")
    #         # This should never happen. The base case in our traversal is reaching a ComposedOp or
    #         # a StateFn. If a leaf of the computational graph is not one of these two, then the
    #         # original operator we are trying to differentiate is not an expectation value or a
    #         # state. Thus it's not clear what the user wants.
    #
    # def get_grad_combo_fn(self, operator: ListOp) -> Callable:
    #     """Get the derivative of the operator combo_fn.
    #
    #     Args:
    #         operator: The operator for whose combo_fn we want to get the gradient.
    #
    #     Returns:
    #         function which evaluates the partial gradient of operator._combo_fn
    #         with respect to each element of operator.oplist
    #
    #     Raises:
    #         Exception: If the operator is a ``ComposedOp``.
    #         Exception: If the gradient combo function could be differentiated.
    #     """
    #     # TODO commented
    #     if isinstance(operator, ComposedOp):
    #         # TODO don't raise bare exception, use e.g. TypeError or AquaError
    #         raise Exception("FIGURE OUT HOW THIS CODE WAS REACHED")
    #
    #     n = len(operator.oplist)
    #     indices = list(range(n))
    #
    #
    #     # jax needs the combo_fn to have n inputs, rather than a list of n inputs
    #     def wrapped_combo_fn(*x):
    #         return operator._combo_fn(list(x))
    #     try:
    #         # import jax.numpy as np
    #         grad_combo_fn = jit(grad(wrapped_combo_fn, indices))
    #         # grad_combo_fn = jit(grad(operator._combo_fn))
    #         # return grad_combo_fn
    #         # TODO remove this!
    #         # Test to see if the grad function breaks for a trivial input
    #         # grad_combo_fn(*([0] * n))
    #     except Exception:
    #         # TODO don't raise bare exception!!
    #         raise Exception("An error occurred when attempting to differentiate a combo_fn")
    #
    #
    #
    #     # ops will be the concatenation of the original oplist with the gradients
    #     # of each operator in the original oplist.
    #     # If your original ListOp contains k elements, then ops will have length 2k.
    #     def chain_rule_combo_fn(ops, grad_combo_fn):
    #         # Get the first half of the values in ops and convert them to floats (or jax breaks)
    #         opvals = [np.float(np.real(val)) for val in ops[:int(len(ops)/2)]]
    #         # Get the derivatives of each op in oplist w.r.t the relevant parameter
    #         derivs = [np.float(np.real(val)) for val in ops[int(len(ops)/2):]]
    #         # Get the partial derivatives of the combo_fn with respect to each op in oplist
    #         pderivs = [partial.tolist() for partial in grad_combo_fn(*opvals)]
    #         # return the dot product to compute the final derivative of the operator with
    #         # respect to the specified parameter.
    #         return np.dot(pderivs, derivs)
    #
    #     return partial(chain_rule_combo_fn, grad_combo_fn=grad_combo_fn)
    # ---------------------------------------------------------------------

    def _get_gates_for_param(self,
                             param: ParameterExpression,
                             qc: QuantumCircuit) -> List[Instruction]:
        # Check if a parameter is used more often than once in a quantum circuit and return a list
        # of quantum circuits which enable independent adding of pi/2 factors without affecting all
        # gates in which the parameters is used.

        # TODO check if param appears in multiple gates of the quantum circuit.
        # TODO deepcopy qc and replace the parameters by independent parameters such that they can
        # be shifted independently by pi/2
        return qc._parameter_table[param]
