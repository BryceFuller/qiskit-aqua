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

"""The module for Quantum the Fisher Information."""

from typing import Optional, Tuple, List, Dict, Union
import warnings

import numpy as np
from copy import deepcopy

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter, ParameterVector

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z


from qiskit.extensions.standard import HGate, XGate, SdgGate, SGate, ZGate

from qiskit.aqua import QuantumInstance

from qiskit.aqua.operators.gradients import GradientBase


class QFI(GradientBase):
    r"""Compute the Quantum Fisher Information given a pure, parametrized quantum state.
        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 0.25.
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
        Returns
            ListOp[ListOp] where the operator at position k,l corresponds to [QFI]kl
        """
        # TODO choose for which parameters we want the QFI
        # TODO integrate diagonal without ancilla
        if isinstance(operator, ListOp):
            for op in operator.oplist:
                # TODO traverse through operator and get the states to compute the QFI for
                # TODO inplace
                # TODO Do this for every independent circuit, rest of product rule to be handled here
                op = self._ancilla_grad_states(op) # TODO change this inplace

            # TODO iterate through params and check if in op - create list/dict to store the params locations
        else:
            operator = self._ancilla_grad_states(operator) # change this inplace

        return operator

    def _ancilla_qfi(self, op: OperatorBase) -> ListOp:
        """Generate the operators whose evaluation leads to the full QFI.

        Args:
            op: The operator representing the quantum state for which we compute the QFI.

        Returns:
            Operators which give the QFI.
            If a parameter appears multiple times, one circuit is created per parameterized gates to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        # QFI & phase fix observable
        qfi_observable = 4 * ((I ^ op.num_qubits) ^ Z - op ^ Z)
        # phase_fix_observable = (I ^ op.num_qubits) ^ (X + 1j * Y)  # see https://arxiv.org/pdf/quant-ph/0108146.pdf
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        qfi_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        qfi_gates = {}
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
            params.append(param)
            gates_to_parameters[param] = []
            qfi_coeffs[param] = []
            qfi_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    qfi_coeffs[param].append(c_g[0])
                    qfi_gates[param].append(c_g[1])

        qfi_operators = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        additional_qubits = ([ancilla], [])
        # create a copy of the original circuit with an additional ancilla register
        circuit = QuantumCircuit(*state_qc.qregs, qr_ancilla)
        circuit.data = state_qc.data
        params = list(gates_to_parameters.keys())
        # apply Hadamard on ancilla
        self.insert_gate(circuit, gates_to_parameters[params[0]][0], HGate(),
                    qubits=[ancilla])
        # Get the circuits needed to compute A_ij
        for i in range(len(params)): #loop over parameters
            qfi_ops = []
            # TODO Check if this overhead can be reduced or is cached by/with the OpFlow
            # j = 0
            # while j <= i: #loop over parameters
            for j in range(len(params)):

                # construct the circuits
                for m, gates_to_insert_i in enumerate(qfi_gates[params[i]]):
                    for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                        coeff_i = qfi_coeffs[params[i]][m][k]
                        for n, gates_to_insert_j in enumerate(qfi_gates[params[j]]):
                            for l, gate_to_insert_j in enumerate(gates_to_insert_j):
                                coeff_j = qfi_coeffs[params[j]][n][l]
                                # create a copy of the original circuit with the same registers
                                qfi_circuit = QuantumCircuit(*circuit.qregs)
                                qfi_circuit.data = circuit.data

                                # Fix ancilla phase
                                sign = np.sign(np.conj(coeff_i)*coeff_j)
                                complex = np.iscomplex(np.conj(coeff_i)*coeff_j)
                                if sign == -1:
                                    if complex:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], SdgGate(),
                                                    qubits=[ancilla])
                                    else:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], ZGate(),
                                                    qubits=[ancilla])
                                else:
                                    if complex:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], SGate(),
                                                    qubits=[ancilla])

                                self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], XGate(),
                                            qubits=[ancilla])

                                # Insert controlled, intercepting gate - controlled by |1>
                                self.insert_gate(qfi_circuit, gates_to_parameters[params[i]][m], gate_to_insert_i,
                                                                         additional_qubits=additional_qubits)

                                self.insert_gate(qfi_circuit, gate_to_insert_i, XGate(), qubits=[ancilla], after=True)

                                # Insert controlled, intercepting gate - controlled by |0>
                                self.insert_gate(qfi_circuit, gates_to_parameters[params[j]][n], gate_to_insert_j,
                                                                         additional_qubits=additional_qubits)

                                '''TODO check if we could use the trimming 
                                What speaks against it is the new way to compute the phase fix directly within 
                                the observable here the trimming wouldn't work. The other way would be more efficient
                                in terms of computation but this is more convenient to write it.'''

                                # Remove redundant gates
                                # qfi_circuit = self.trim_circuit(qfi_circuit, gates_to_parameters[params[i]][m])

                                qfi_circuit.h(ancilla)
                                if m == 0 and k == 0:
                                    qfi_op = [np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(qfi_circuit)]
                                else:
                                    qfi_op += np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(qfi_circuit)
                qfi_ops += [qfi_op]
            qfi_operators.append(qfi_ops)
        return ~qfi_observable @ ListOp(qfi_operators)
