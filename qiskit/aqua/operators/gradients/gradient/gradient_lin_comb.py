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

"""The module to compute the state gradient with the linear combination method."""
from collections.abc import Iterable
from typing import Optional, Union, List
from functools import partial
from copy import deepcopy
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import ZGate, SGate, SdgGate, HGate

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, ComposedOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.aqua.operators.operator_globals import Z, I, One, Zero
from qiskit.quantum_info import partial_trace
from ..gradient_base import GradientBase


class GradientLinComb(GradientBase):
    """Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω using the linear combination method.
    respectively the gradients of the sampling probabilities of the basis states of a state |ψ(ω)〉w.r.t. ω.
    This method employs a linear combination of unitaries, see e.g. https://arxiv.org/pdf/1811.11184.pdf
    """

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
                ) -> OperatorBase:
        """
        Args:
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """


        # self._meas_op = None
        # for op in operator.oplist:
        #     if op.is_measurement:
        #         measurement = Z ^ deepcopy(op)
        #     else:
        #         state = deepcopy(op)
        # if isinstance(operator, ComposedOp):
        #     measurement = operator[0]
        #     state = operator[-1]
        #

        # self._params = params
        # self._operator_has_measurement = False
        return self._prepare_operator(operator, params)
    #     self._params = params
    #     self._gradient_operator = operator
    #
    #     return self._get_grad_states(operator)
    #
    # def _get_measurement(self, operator):
    #     if isinstance(operator, ListOp):
    #         return operator.traverse(self._get_measurement)
    #     elif operator.is_measurement:
    #         return Z ^ operator
    #     else:
    #         return None
    #
    # def _get_grad_states(self, operator):
    #     if isinstance(operator, (CircuitStateFn, CircuitOp)):
    #         return self._grad_states(operator, self._params)
    #     elif isinstance(operator, ListOp):
    #         return operator.traverse(self._get_grad_states)
    #     else:
    #         raise TypeError('Please define an operator that incorporates a CircuitStateFn.')

    # def _prepare_operator(self, operator):
    #     if isinstance(operator, ListOp):
    #         return operator.traverse(self._prepare_operator)
    #     elif isinstance(operator, StateFn):
    #         if operator.is_measurement:
    #             self._operator_has_measurement = True
    #             return operator.traverse(self._prepare_operator)
    #     elif isinstance(operator, PrimitiveOp):
    #         return Z ^ operator
    #     if isinstance(operator, (CircuitStateFn, CircuitOp)):
    #         return self._grad_states(operator, self._params)
    #     return operator

    def _prepare_operator(self, operator, params):
        if isinstance(operator, ComposedOp):
            if not isinstance(operator[0], StateFn) or not operator[0]._is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if not isinstance(operator[-1], StateFn) or operator[-1]._is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if operator[0].is_measurement:
                if len(operator.oplist) == 2:
                    state_op = operator[1]
                    return self._grad_states(state_op, meas_op=(~StateFn(Z) ^ operator[0]),
                                                     target_params=params)
                else:
                    state_op = deepcopy(operator)
                    state_op.oplist.pop(0)
                    return state_op.traverse(partial(self._grad_states, meas_op=(~StateFn(Z) ^ operator[0]),
                                                 target_params=params))

            else:
                return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                self._operator_has_measurement = True
                return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, PrimitiveOp):
            return operator
        elif isinstance(operator, (CircuitStateFn, CircuitOp)):
            return self._grad_states(operator, target_params=params)
        return operator

    def _grad_states(self,
                     state_op: OperatorBase,
                     meas_op: Optional[OperatorBase] = None,
                     target_params: Optional[Union[Parameter, ParameterVector, List]] = None
                     ) -> ListOp:
        """Generate the gradient states.

        Args:
            state_op: The operator representing the quantum state for which we compute the gradient.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
        """
        # Get measurement operator
        # meas_op = self._get_measurement(self._gradient_operator)

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the grad for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the grad for every parameter
        grad_gates = {}
        state_qc = deepcopy(state_op.primitive)
        if not isinstance(target_params, Iterable):
            target_params = [target_params]
        for param in target_params:
            elements = state_qc._parameter_table[param]
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized
                # gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                c = []
                g = []
                for j, gate_param in enumerate(element[0].params):
                    c.extend(coeffs_gates[j][0])
                    g.extend(coeffs_gates[j][1])
                grad_coeffs[param].append(c)
                grad_gates[param].append(g)
        if len(target_params) > 1:
            states = []
        qr_work = QuantumRegister(1, 'work_qubit')
        work_q = qr_work[0]
        additional_qubits = ([work_q], [])
        # create a copy of the original state with an additional work_q register
        # Get the states needed to compute the gradient
        for param in target_params:  # loop over parameters
            # construct the states
            for m, gates_to_insert_i in enumerate(grad_gates[param]):
                for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                    grad_state = QuantumCircuit(*state_qc.qregs, qr_work)
                    grad_state.compose(state_qc, inplace=True)

                    # apply Hadamard on work_q
                    self.insert_gate(
                        grad_state, gates_to_parameters[target_params[0]][0], HGate(), qubits=[work_q]
                    )

                    # Fix work_q phase
                    coeff_i = grad_coeffs[param][m][k]
                    sign = np.sign(coeff_i)
                    is_complex = np.iscomplex(coeff_i)
                    if sign == -1:
                        if is_complex:
                            self.insert_gate(grad_state, gates_to_parameters[target_params[0]][0],
                                             SdgGate(), qubits=[work_q])
                        else:
                            self.insert_gate(grad_state, gates_to_parameters[target_params[0]][0],
                                             ZGate(), qubits=[work_q])
                    else:
                        if is_complex:
                            self.insert_gate(grad_state, gates_to_parameters[target_params[0]][0],
                                             SGate(), qubits=[work_q])

                    # Insert controlled, intercepting gate - controlled by |0>
                    self.insert_gate(grad_state, gates_to_parameters[param][m],
                                     gate_to_insert_i,
                                     additional_qubits=additional_qubits)
                    grad_state.h(work_q)

                    state = np.sqrt(np.abs(coeff_i) * 2) * CircuitStateFn(grad_state)
                    # Chain Rule parameter expressions
                    gate_param = gates_to_parameters[param][m].params[k]
                    if meas_op:
                        if gate_param == param:
                            state = meas_op @ state
                        else:
                            if isinstance(gate_param, ParameterExpression):
                                import sympy as sy
                                expr_grad = self.parameter_expression_grad(gate_param, param)
                                # Square root needed bc the coefficients are squared in the expectation value
                                # TODO enable complex parameter expressions
                                # expr_grad._symbol_expr = sy.sqrt(expr_grad._symbol_expr)
                                state = (expr_grad * meas_op) @ state
                            else:
                                state = ~StateFn(One) @ Zero
                    # if meas_op:
                    #     if gate_param == param:
                    #         state = meas_op @ state
                    #     else:
                    #         if isinstance(gate_param, ParameterExpression):
                    #             import sympy as sy
                    #             expr_grad = self.parameter_expression_grad(gate_param, param)
                    #             # Square root needed bc the coefficients are squared in the expectation value
                    #             # TODO enable complex parameter expressions
                    #             # expr_grad._symbol_expr = sy.sqrt(expr_grad._symbol_expr)
                    #             state = expr_grad * meas_op @ state
                    #         else:
                    #             state = ~StateFn(One) @ Zero
                    else:
                        def combo_fn(x):
                            # TODO parameter expression
                            x = x.primitive
                            # Generate the operator which computes the linear combination
                            lin_comb_op = (I ^ state_op.num_qubits) ^ Z
                            lin_comb_op = lin_comb_op.to_matrix()
                            # Compute a partial trace over the working qubit needed to compute the linear combination
                            if isinstance(x, list) or isinstance(x, np.ndarray):
                                # TODO check if output is prob or sv - in case of prob get rid of np.dot
                                return [np.diag(partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0]).data)
                                        for item in x]
                            else:
                                # TODO check if output is prob or sv - in case of prob get rid of np.dot
                                return np.diag(partial_trace(lin_comb_op.dot(np.outer(x, np.conj(x))), [0]).data)
                        state = ListOp(state, combo_fn=combo_fn)

                    if m == 0 and k == 0:
                        op = state
                    else:
                        # Product Rule
                        op += state
            # The division is necessary to compensate for normalization of summed StateFns
            if len(target_params) > 1:
                states += [op]
            # states += [state_op / np.sqrt(len(gates_to_parameters[param]))]
        if len(target_params) > 1:
            return ListOp(states) * state_op.coeff
        else:
            return op * state_op.coeff
