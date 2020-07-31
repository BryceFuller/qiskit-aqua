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
from qiskit.aqua.operators.gradients.gradient_base import GradientBase
from qiskit.aqua.operators import OperatorBase, ListOp

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

        # TODO: Check if operator includes a state else throw an error/ or warning?
        measurement = False
        state_given = False
        if isinstance(operator, ListOp):
            for op in operator.oplist:
                # TODO check which param is in which op - create list/dict to store the params locations.
                # TODO Then the gradients must be computed for the different ops and summed up accordingly.
                if op.is_measurement:
                    measurement = True
                else:
                    state_given = True
        else:
            if not operator.is_measurement:
                state_given = True

        if not state_given:
            raise TypeError('Currently the gradient framework only supports gradient evaluation with respect to '
                            'expectation values and sampling probabilities of quantum states. '
                            'Please define an operator which includes a quantum state.')

        if measurement:
            # TODO: if params in observable return observable_gradient else return state_gradient depending on method
            pass
        else:
            # TODO: return probability gradient
            pass

    # TODO get ParameterExpression in the different gradients
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
