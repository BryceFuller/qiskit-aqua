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

""" StateHessian Class """
from collections.abc import Iterable
from typing import Optional, Union, List, Tuple
from copy import deepcopy
from functools import partial
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import HGate, SGate, SdgGate, ZGate
from qiskit.quantum_info import partial_trace

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, ComposedOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import I, Z, One, Zero

from ..gradient_base import GradientBase


class HessianLinComb(GradientBase):
    """Compute the state hessian using the linear combination method."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, List[Parameter], Tuple[Parameter, Parameter],
                               List[Tuple[Parameter, Parameter]]]] = None
                ) -> OperatorBase:
        """
        Args:
            operator:The operator corresponding to our quantum state we are taking the gradient
                of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to
            d^2⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω_kdω_l
        """
        return self._prepare_operator(operator, params)

    def _prepare_operator(self, operator, params):
        if isinstance(operator, ComposedOp):
            if not isinstance(operator[0], StateFn) or not operator[0]._is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if not isinstance(operator[-1], StateFn) or operator[-1]._is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if operator[0].is_measurement:
                if len(operator.oplist) == 2:
                    state_op = operator[1]
                    return self._hessian_states(state_op, meas_op=4 * (~StateFn(Z ^ I) ^ operator[0]),
                                             target_params=params)
                else:
                    state_op = deepcopy(operator)
                    state_op.oplist.pop(0)
                    return state_op.traverse(partial(self._hessian_states, meas_op=(~StateFn(Z) ^ operator[0]),
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
            return self._hessian_states(operator, target_params=params)
        return operator

    def _hessian_states(self,
                        state_op: OperatorBase,
                        meas_op: Optional[OperatorBase] = None,
                        target_params: Optional[Union[Tuple[Parameter, Parameter], List[Tuple[Parameter, Parameter]]]]
                        = None) -> OperatorBase:
        """Generate the operators whose evaluation leads to the full QFI.

        Args:
            state_op: The operator representing the quantum state for which we compute the hessian.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are computing the hessian wrt: ω

        Returns:
            Operators which give the hessian. If a parameter appears multiple times, one circuit is
            created per parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If ``operator`` is of unsupported type.
        """

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the hessian for every parameter
        hessian_coeffs = {}
        # Dictionary which relates the gates needed for the hessian for every parameter
        hessian_gates = {}
        # Get the quantum circuit corresponding to the state operator
        state_qc = deepcopy(state_op.primitive)
        if isinstance(target_params, list) and isinstance(target_params[0], tuple):
            tuples_list = deepcopy(target_params)
            target_params = []
            for tuples in tuples_list:
                for param in tuples:
                    if param not in target_params:
                        target_params.append(param)
        else:
            raise TypeError('Please define in the parameters for which the Hessian is evaluated either '
                            'as parameter tuple or a list of parameter tuples')

        for param in target_params:
            elements = state_qc._parameter_table[param]
            gates_to_parameters[param] = []
            hessian_coeffs[param] = []
            hessian_gates[param] = []
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
                    hessian_coeffs[param].append(c)
                    hessian_gates[param].append(g)

        hessian_operators = []
        qr_add0 = QuantumRegister(1, 'work_qubit0')
        work_q0 = qr_add0[0]
        qr_add1 = QuantumRegister(1, 'work_qubit1')
        work_q1 = qr_add1[0]
        # create a copy of the original circuit with an additional working qubit register
        circuit = QuantumCircuit(*state_qc.qregs, qr_add0, qr_add1)
        circuit.data = state_qc.data
        # apply Hadamard on ancilla
        self.insert_gate(circuit, gates_to_parameters[target_params[0]][0], HGate(),
                         qubits=[work_q0])
        self.insert_gate(circuit, gates_to_parameters[target_params[0]][0], HGate(),
                         qubits=[work_q1])
        # Get the circuits needed to compute A_ij
        hessian_ops = []
        for param_a, param_b in tuples_list:
            for i, gates_to_insert_a in enumerate(hessian_gates[param_a]):
                for j, gate_to_insert_a in enumerate(gates_to_insert_a):
                    coeff_a = hessian_coeffs[param_a][i][j]
                    hessian_circuit_temp = QuantumCircuit(*circuit.qregs)
                    hessian_circuit_temp.data = circuit.data
                    # Fix working qubit 0 phase
                    sign = np.sign(coeff_a)
                    is_complex = np.iscomplex(coeff_a)
                    if sign == -1:
                        if is_complex:
                            self.insert_gate(hessian_circuit_temp,
                                             gates_to_parameters[target_params[0]][0],
                                             SdgGate(),
                                             qubits=[work_q0])
                        else:
                            self.insert_gate(hessian_circuit_temp,
                                             gates_to_parameters[target_params[0]][0],
                                             ZGate(),
                                             qubits=[work_q0])
                    else:
                        if is_complex:
                            self.insert_gate(hessian_circuit_temp,
                                             gates_to_parameters[target_params[0]][0],
                                             SGate(),
                                             qubits=[work_q0])

                    # Insert controlled, intercepting gate - controlled by |1>
                    self.insert_gate(hessian_circuit_temp, gates_to_parameters[param_a][i],
                                     gate_to_insert_a, additional_qubits=([work_q0], []))

                    for m, gates_to_insert_b in enumerate(hessian_gates[param_b]):
                        for n, gate_to_insert_b in enumerate(gates_to_insert_b):
                            coeff_b = hessian_coeffs[param_b][m][n]
                            # create a copy of the original circuit with the same registers
                            hessian_circuit = QuantumCircuit(*hessian_circuit_temp.qregs)
                            hessian_circuit.data = hessian_circuit_temp.data

                            # Fix working qubit 1 phase
                            sign = np.sign(coeff_b)
                            is_complex = np.iscomplex(coeff_b)
                            if sign == -1:
                                if is_complex:
                                    self.insert_gate(hessian_circuit,
                                                     gates_to_parameters[target_params[0]][0],
                                                     SdgGate(),
                                                     qubits=[work_q1])
                                else:
                                    self.insert_gate(hessian_circuit,
                                                     gates_to_parameters[target_params[0]][0],
                                                     ZGate(),
                                                     qubits=[work_q1])
                            else:
                                if is_complex:
                                    self.insert_gate(hessian_circuit,
                                                     gates_to_parameters[target_params[0]][0],
                                                     SGate(),
                                                     qubits=[work_q1])

                            # Insert controlled, intercepting gate - controlled by |1>
                            self.insert_gate(hessian_circuit,
                                             gates_to_parameters[param_b][m],
                                             gate_to_insert_b,
                                             additional_qubits=([work_q1], []))

                            hessian_circuit.h(work_q0)
                            hessian_circuit.cz(work_q1, work_q0)
                            hessian_circuit.h(work_q1)

                            term = state_op.coeff * np.sqrt(np.abs(coeff_a) * np.abs(coeff_b)) * \
                                   CircuitStateFn(hessian_circuit)

                            # Chain Rule Parameter Expression
                            gate_param_a = gates_to_parameters[param_a][i].params[j]
                            gate_param_b = gates_to_parameters[param_b][m].params[n]

                            if meas_op:
                                meas = deepcopy(meas_op)
                                if isinstance(gate_param_a, ParameterExpression):
                                    import sympy as sy
                                    expr_grad = self.parameter_expression_grad(gate_param_a, param_a)
                                    meas *= expr_grad
                                if isinstance(gate_param_b, ParameterExpression):
                                    import sympy as sy
                                    expr_grad = self.parameter_expression_grad(gate_param_a, param_a)
                                    meas *= expr_grad
                                term = meas @ term

                            else:
                                def combo_fn(x):

                                    x = x.primitive
                                    # Generate the operator which computes the linear combination
                                    lin_comb_op = (I ^ state_op.num_qubits) ^ Z
                                    lin_comb_op = lin_comb_op.to_matrix()
                                    # Compute a partial trace over the working qubit needed to compute the
                                    # linear combination
                                    if isinstance(x, list) or isinstance(x, np.ndarray):
                                        # TODO check if output is prob or sv - in case of prob get rid of np.dot
                                        return [np.diag(
                                            partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0]).data)
                                            for item in x]
                                    else:
                                        # TODO check if output is prob or sv - in case of prob get rid of np.dot
                                        return np.diag(
                                            partial_trace(lin_comb_op.dot(np.outer(x, np.conj(x))), [0]).data)

                                # TODO parameter expression

                                term = ListOp(term, combo_fn=combo_fn)

                            if i == 0 and j == 0 and m == 0 and n == 0:
                                hessian_op = term
                            else:
                                # Product Rule
                                hessian_op += term
            # Create a list of Hessian elements w.r.t. the given parameter tuples
            if len(tuples_list) == 1:
                return hessian_op
            else:
                hessian_ops += [hessian_op]
        return ListOp(hessian_ops)
