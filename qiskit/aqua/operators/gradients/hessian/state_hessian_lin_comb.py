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

from typing import Optional, Callable, Union, List, Dict
import logging
from functools import partial, reduce
from copy import deepcopy
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Gate

from qiskit.aqua.operators import OperatorBase, ListOp, PauliOp, CircuitOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z
from qiskit.aqua.operators.expectations import PauliExpectation
from ..gradient_base import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.circuit.library.standard_gates import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, \
    CZGate, \
    U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, XGate, \
    SdgGate, SGate, ZGate

logger = logging.getLogger(__name__)


class StateHessianAncilla(GradientBase):
    r"""
    We are interested in computing:
    d^2⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω_kdω_l  for ω in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
        Returns
            ListOp[ListOp] where the operator at position k,l corresponds to d^2⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω_kdω_l
        """

        return self._prepare_operator(operator)

    def _prepare_operator(self, operator, params):
        if isinstance(operator, ListOp):
            return operator.traverse(self.prepare_operator)
        elif isinstance(operator, StateFn):
            if operator.is_measurement == True:
                return operator.traverse(self.prepare_operator)
        elif isinstance(operator, PrimitiveOp):
            return 2 * (operator ^ Z ^ Z)
        if isinstance(operator, (QuantumCircuit, CircuitStateFn, CircuitOp)):
            # operator.primitive.add_register(QuantumRegister(1, name="ancilla"))
            operator = self._ancilla_qfi(operator, params)
        return operator

    def _hessian_states(self, op: OperatorBase,
                     target_params: Union[Parameter, ParameterVector, List] = None) -> ListOp:
        """Generate the operators whose evaluation leads to the full QFI.

        Args:
            op: The operator representing the quantum state for which we compute the hessian.
            target_params: The parameters we are computing the hessian wrt: ω

        Returns:
            Operators which give the hessian.
            If a parameter appears multiple times, one circuit is created per parameterized gates to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the hessian for every parameter
        hessian_coeffs = {}
        # Dictionary which relates the gates needed for the hessian for every parameter
        hessian_gates = {}
        # Loop throuh the parameters in the circuit
        params = []

        if isinstance(op, CircuitStateFn) or isinstance(op, CircuitOp):
            pass
        elif isinstance(op, DictStateFn) or isinstance(op, VectorStateFn):
            op = DictToCircuitSum.convert(op)  # Todo inplace
        else:
            raise TypeError('Ancilla gradients only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        state_qc = deepcopy(op.primitive)
        for param, elements in state_qc._parameter_table.items():
            # TODO param expressions
            if param not in target_params:
                continue
            if param not in params:
                params.append(param)
                gates_to_parameters[param] = []
            hessian_coeffs[param] = []
            hessian_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    hessian_coeffs[param].append(c_g[0])
                    hessian_gates[param].append(c_g[1])

        hessian_operators = []
        qr_add0 = QuantumRegister(1, 'work_qubit0')
        work_q0 = qr_add0[0]
        qr_add1 = QuantumRegister(1, 'work_qubit1')
        work_q1 = qr_add1[0]
        # create a copy of the original circuit with an additional ancilla register
        circuit = QuantumCircuit(*state_qc.qregs, qr_add0, qr_add1)
        circuit.data = state_qc.data
        # params = list(gates_to_parameters.keys())
        # apply Hadamard on ancilla
        self.insert_gate(circuit, gates_to_parameters[params[0]][0], HGate(),
                    qubits=[work_q0])
        # Get the circuits needed to compute A_ij
        for i in range(len(params)): #loop over parameters
            hessian_ops = []
            # TODO Check if this overhead can be reduced or is cached by/with the OpFlow
            # j = 0
            # while j <= i: #loop over parameters
            for j in range(len(params)):

                # construct the circuits
                for m, gates_to_insert_i in enumerate(hessian_gates[params[i]]):
                    for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                        coeff_i = hessian_coeffs[params[i]][m][k]
                        hessian_circuit_temp = QuantumCircuit(*circuit.qregs)
                        hessian_circuit_temp.data = circuit.data
                        # Fix working qubit 0 phase
                        sign = np.sign(coeff_i)
                        complex = np.iscomplex(coeff_i)
                        if sign == -1:
                            if complex:
                                self.insert_gate(hessian_circuit_temp, gates_to_parameters[params[0]][0], SdgGate(),
                                                 qubits=[work_q0])
                            else:
                                self.insert_gate(hessian_circuit_temp, gates_to_parameters[params[0]][0], ZGate(),
                                                 qubits=[work_q0])
                        else:
                            if complex:
                                self.insert_gate(hessian_circuit_temp, gates_to_parameters[params[0]][0], SGate(),
                                                 qubits=[work_q0])

                        # Insert controlled, intercepting gate - controlled by |1>
                        self.insert_gate(hessian_circuit_temp, gates_to_parameters[params[i]][m], gate_to_insert_i,
                                         additional_qubits=([work_q0], []))

                        for n, gates_to_insert_j in enumerate(hessian_gates[params[j]]):
                            for l, gate_to_insert_j in enumerate(gates_to_insert_j):
                                coeff_j = hessian_coeffs[params[j]][n][l]
                                # create a copy of the original circuit with the same registers
                                hessian_circuit = QuantumCircuit(*hessian_circuit_temp.qregs)
                                hessian_circuit.data = hessian_circuit_temp.data

                                # Fix working qubit 1 phase
                                # Fix ancilla phase
                                sign = np.sign(coeff_j)
                                complex = np.iscomplex(coeff_j)
                                if sign == -1:
                                    if complex:
                                        self.insert_gate(hessian_circuit, gates_to_parameters[params[0]][0],
                                                         SdgGate(),
                                                         qubits=[work_q1])
                                    else:
                                        self.insert_gate(hessian_circuit, gates_to_parameters[params[0]][0],
                                                         ZGate(),
                                                         qubits=[work_q1])
                                else:
                                    if complex:
                                        self.insert_gate(hessian_circuit, gates_to_parameters[params[0]][0],
                                                         SGate(),
                                                         qubits=[work_q1])

                                # Insert controlled, intercepting gate - controlled by |1>
                                self.insert_gate(hessian_circuit, gates_to_parameters[params[j]][n], gate_to_insert_j,
                                                                         additional_qubits=([work_q1], []))

                                hessian_circuit.cz(work_q1, work_q0)
                                hessian_circuit.h(work_q0)
                                hessian_circuit.h(work_q1)
                                if m == 0 and k == 0:
                                    hessian_op = [np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(hessian_circuit)]
                                else:
                                    hessian_op += np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(hessian_circuit)
                hessian_ops += [hessian_op]
            hessian_operators.append(hessian_ops)
        return ListOp(hessian_operators)
