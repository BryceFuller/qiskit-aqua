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

from typing import Optional, Union, Tuple, List
import sympy as sy

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from ..gradient_base import GradientBase
from qiskit.aqua.operators import OperatorBase, ListOp


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
        params: Optional[List] = None,
        method: str = 'param_shift',
        natural_gradient: bool = False) -> OperatorBase:

        r"""
        Args:
            operator: The measurement operator we are taking the gradient of
            state_operator:  The operator corresponding to our state preparation circuit
            parameters: The parameters we are taking the gradient with respect to
            method: The method used to compute the state/probability gradient. ['param_shift', 'ancilla']
                    Deprecated for observable gradient
        Returns:
            gradient_operator: An operator whose evaluation yields the Gradient
        """
        self._operator = operator
        self._params = params

        # Prepare operator

        grads = []
        if len(params) != len(set(params)):
            raise TypeError('Please provide an array that consists of unique parameter items.')
        for param in params:
            param_grad = None
            if isinstance(operator, ListOp):
                pass
                # get grad_combo_fn = operator.combo_fn
                # Check if sympy function
                # Else try jax
                # traverse through operators - recursive
                for op_param in operator.primitive.params:
                    if isinstance(op_param, ParameterExpression):
                        if param in op_param.parameters:
                            param_expr_grad = sy.Derivative(op_param, param)
                            if operator.is_measurement:
                                # Check for the corresponding state and compute observable_gradient
        #                         raise TypeError(
    #                             'Currently the gradient framework only supports gradient evaluation with respect to '
    #                             'expectation values and sampling probabilities of quantum states. '
    #                             'Please define an operator which includes a quantum state.')
                                p_grad = 0
                            else:
                                # Check if the state operator is part of an expectation value and compute either
                                # state_gradient or probability_gradient
                                p_grad = 0
                            if not param_grad:
                                param_grad = p_grad * param_expr_grad
                            else:
                                param_grad += p_grad * param_expr_grad # TODO this should respect the comboFn!!! not only plus what if param in observable and state
                    else:
                        if param == op_param:
                            if operator.is_measurement:
                                # Check for the corresponding state and compute observable_gradient
        #                         raise TypeError(
    #                             'Currently the gradient framework only supports gradient evaluation with respect to '
    #                             'expectation values and sampling probabilities of quantum states. '
    #                             'Please define an operator which includes a quantum state.')
                                p_grad = 0
                            else:
                                # Check if the state operator is part of an expectation value and compute either
                                # state_gradient or probability_gradient
                                p_grad = 0
                            if not param_grad:
                                param_grad = p_grad
                            else:
                                param_grad += p_grad # TODO this should respect the comboFn!!! not only plus what if param in observable and state

            grads.append(param_grad)
        return ListOp(grads)


    # Working title
    def _chain_rule_wrapper_sympy_grad(self,
                                  param: ParameterExpression) -> List[Union[sy.Expr, float]]:
        """
        Get the derivative of a parameter expression w.r.t. the underlying parameter keys
        :param param: Parameter Expression
        :return: List of derivatives of the parameter expression w.r.t. all keys
        """
        expr = param._symbol_expr
        keys = param._parameter_symbols.keys()
        grad = []
        for key in keys:
            grad.append(sy.Derivative(expr, key).doit())
        return grad

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
