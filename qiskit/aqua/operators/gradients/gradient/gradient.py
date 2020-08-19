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

from typing import Optional, Union, Tuple, List, Callable
from functools import reduce, partial
import numpy as np
from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Parameter, ParameterVector, Instruction
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from ..gradient_base import GradientBase
from qiskit.aqua.operators.gradients.gradient.operator_gradient import ObservableGradient
# from qiskit.aqua.operators.gradients.gradient.prob_gradient import ProbabilityGradient
from qiskit.aqua.operators.gradients.gradient.state_gradient import StateGradient
from qiskit.aqua.operators import OperatorBase, ListOp, SummedOp, ComposedOp, TensoredOp



"""
Structure:
    - grads = []
    - For param in params:
        - Get Grad of ComboFn (trivial if sum or Identity)
        - If operator has measurement 
            - If param in state (potentially within a parameter expression)
                - StateGradient
            - Else OperatorGradient (potentially within a parameter expression)
        - Else ProbabilityGradient (potentially within a parameter expression)
        
        - If param was in param_expr:
            - grads.append(dOperator/d_param_expr * d_param_expr)
        - Else grads.append(dOperator/d_param)
        
    Return ListOp[grads]
        
"""

class Gradient(GradientBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields the first-order gradient with respect to the circuit parameters.
    """

    # pylint: disable=too-many-return-statements
    def convert(self,
        operator: OperatorBase = None,
        # grad_combo_fn: Callable = lambda x : x,
        params: Union[ParameterVector, Parameter] = None,
        method: str = 'param_shift') -> OperatorBase:

        r"""
        Args:
            operator: The operator we are taking the gradient of
            grad_combo_fn: Gradient for a custom operator combo_fn. The gradient for a standard ListOp or sympy
            combo_fn is automatically computed.
            parameters: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient. ['param_shift', 'lin_comb']
                    Deprecated for observable gradient
        Returns:
            gradient_operator: An operator whose evaluation yields the Gradient
        """

        self._operator = operator
        self._params = params
        self._method = method

        # Prepare operator

        grads = []
        if len(params) != len(set(params)):
            raise TypeError('Please provide an array that consists of unique parameter items.')
        for param in params:
            param_grad = []
            if isinstance(operator, ListOp):
                grad_combo_fn = self._get_grad_combo_fn(operator, grad_combo_fn)
                # traverse through operators - recursive
                for op in operator:
                    param_grad.append(self._get_param_grads(op, param))
            else:
                param_grad.append(self._get_param_grads(operator, param))
                # Recombine param_grad according to the operator's combo_fn
            grads.append(ListOp(param_grad, combo_fn=grad_combo_fn))
        return ListOp(grads)

    def _get_param_grads(self,
                         op: OperatorBase,
                         param: Union[ParameterVector, Parameter]) -> OperatorBase:
        """
        For a given operator and parameter, check if the parameter is used in the operator and if so compute the
        respective gradient
        Args:
            op: The operator we are taking the gradient of
            param: The param we are taking the gradient with respect to

        Returns:
            OperatorBase: Operator corresponding to the gradient for op and param

        """
        for op_param in op.primitive.params:
            if isinstance(op_param, ParameterExpression):
                if param in op_param.parameters:
                    # TODO stacked parameter expressions
                    param_expr_grad = op_param.diff(param)
                    if op.is_measurement:
                        # Check for the corresponding state and compute observable_gradient
                        #                         raise TypeError(
                        #                             'Currently the gradient framework only supports gradient
                        #                             'evaluation with respect to '
                        #                             'expectation values and sampling probabilities of quantum states.'
                        #                             'Please define an operator which includes a quantum state.')
                        # TODO the op in the next level needs to go one level up i.e. get observable and state
                        return ObservableGradient.convert(op, op_param) * param_expr_grad
                    else:
                        # Check if the state operator is part of an expectation value and compute either
                        # state_gradient or probability_gradient
                        # TODO the op in the next level needs to go one level up i.e. get observable and state
                        return StateGradient.convert(op, op_param, self._method) * param_expr_grad
                        # return ProbabilityGradient.convert(op, op_param, self._method)

            else:
                if param == op_param:
                    if op.is_measurement:
                        # Check for the corresponding state and compute observable_gradient
                        #                         raise TypeError(
                        #                             'Currently the gradient framework only supports gradient
                        #                             'evaluation with respect to '
                        #                             'expectation values and sampling probabilities of quantum states.'
                        #                             'Please define an operator which includes a quantum state.')
                        return ObservableGradient.convert(op, param)
                    else:
                        # Check if the state operator is part of an expectation value and compute either
                        # state_gradient or probability_gradient
                        return StateGradient.convert(op, param, self._method)
                        # return ProbabilityGradient.convert(op, param, self._method)


    def _get_grad_combo_fn(self,
                           operator: ListOp,
                           grad_combo_fn: Callable) -> Callable:
        """
        Get the derivative of the operator combo_fn
        Args:
            operator: The operator for whose combo_fn we want to get the gradient.
            grad_combo_fn: The derivative of the standard combo_fn

        Returns:
            Derivative of the operator combo_fn

        """
        from sympy import Function
        if isinstance(operator, SummedOp) or isinstance(operator, TensoredOp):
            grad_combo_fn = operator.combo_fn
        elif isinstance(operator, ComposedOp):
            def grad_composed_combo_fn(x):
                sum = 0
                for i in range(len(x)):
                    y = deepcopy(operator)
                    y[i] = x[i]
                    sum += partial(reduce, np.dot)(y)
                return sum
            grad_combo_fn = grad_composed_combo_fn
        elif isinstance(operator.combo_fn, Function):
            grad_combo_fn = operator.combo_fn.diff()
        return grad_combo_fn


    # # Working title
    # def _chain_rule_wrapper_sympy_grad(self,
    #                               param: ParameterExpression) -> List[Union[sy.Expr, float]]:
    #     """
    #     Get the derivative of a parameter expression w.r.t. the underlying parameter keys
    #     :param param: Parameter Expression
    #     :return: List of derivatives of the parameter expression w.r.t. all keys
    #     """
    #     expr = param._symbol_expr
    #     keys = param._parameter_symbols.keys()
    #     grad = []
    #     for key in keys:
    #         grad.append(sy.Derivative(expr, key).doit())
    #     return grad

    def _get_gates_for_param(self,
                             param: ParameterExpression,
                             qc: QuantumCircuit) -> List[Instruction]:
        """
        Check if a parameter is used more often than once in a quantum circuit and return a list of quantum circuits
        which enable independent adding of pi/2 factors without affecting all gates in which the parameters is used.
        :param param:
        :param qc:
        :return:
        """
        # TODO check if param appears in multiple gates of the quantum circuit.
        # TODO deepcopy qc and replace the parameters by independent parameters such that they can be shifted
        # independently by pi/2
        return qc._parameter_table[param]
