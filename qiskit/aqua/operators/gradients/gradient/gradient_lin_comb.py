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

from typing import Optional, Union, List
from copy import deepcopy
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import ZGate, SGate, SdgGate, HGate

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.aqua.operators.operator_globals import Z, I
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
        self._params = params
        self._operator_has_measurement = False
        return self._prepare_operator(operator)

    def _prepare_operator(self, operator):
        if isinstance(operator, ListOp):
            return operator.traverse(self._prepare_operator)
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                self._operator_has_measurement = True
                return operator.traverse(self._prepare_operator)
        elif isinstance(operator, PrimitiveOp):
            return 2 * Z ^ operator
        if isinstance(operator, (CircuitStateFn, CircuitOp)):
            operator = self._grad_states(operator, self._params)
        return operator

    def _grad_states(self,
                     op: OperatorBase,
                     target_params: Optional[Union[Parameter, ParameterVector, List]] = None
                     ) -> ListOp:
        """Generate the gradient states.

        Args:
            op: The operator representing the quantum state for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
        """

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the grad for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the grad for every parameter
        grad_gates = {}
        # # Loop throuh the parameters in the circuit
        # params = []
        state_qc = deepcopy(op.primitive)
        # for param, elements in state_qc._parameter_table.items():
        for param in target_params:
            elements = state_qc._parameter_table[param]
            # if param not in target_params:
            #     continue
            # if param not in params:
            #     params.append(param)
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            # grad_param_expr_grad[param] = []
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
                    state = np.abs(coeff_i) * CircuitStateFn(grad_state)
                    # TODO fix coeff propagation in ListOps
                    # state = np.sqrt(np.abs(coeff_i)) * CircuitStateFn(grad_state)
                    # Chain Rule parameter expressions
                    gate_param = gates_to_parameters[param][m].params[k]
                    if gate_param == param:
                        pass
                    else:
                        if isinstance(gate_param, ParameterExpression):
                            import sympy as sy
                            expr_grad = self.parameter_expression_grad(gate_param, param)
                            # Square root needed bc the coefficients are squared in the expectation value
                            # TODO enable complex parameter expressions
                            # expr_grad._symbol_expr = sy.sqrt(expr_grad._symbol_expr)
                            state *= expr_grad
                        else:
                            state *= 0

                    if m == 0 and k == 0:
                        state_op = state
                    else:
                        state_op += state

            states += [state_op]
        if self._operator_has_measurement:
            return ListOp(states) * op.coeff
        else:
            def combo_fn(x):
                # Generate the operator which computes the linear combination
                lin_comb_op = 2 * (I ^ op.num_qubits) ^ Z
                lin_comb_op = lin_comb_op.to_matrix()
                # Compute a partial trace over the working qubit needed to compute the linear combination
                if isinstance(x, list) or isinstance(x, np.ndarray):
                    # TODO check if output is prob or sv - in case of prob get rid of np.dot
                    return [np.diag(partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0]).data)
                            for item in x]
                else:
                    # TODO check if output is prob or sv - in case of prob get rid of np.dot
                    return np.diag(partial_trace(lin_comb_op.dot(np.outer(x, np.conj(x))), [0]).data)

            return ListOp(states, combo_fn=combo_fn) * op.coeff
