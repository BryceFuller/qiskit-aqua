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

from typing import Optional, Tuple, List, Dict
import warnings

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile

from qiskit.circuit import Gate, Qubit, Instruction
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.aqua.operators import PauliOp, X, Y, Z, I, CircuitSampler
from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.gradients import GradientBase

from qiskit.aqua.utils.run_circuits import find_regs_by_name

from qiskit.extensions.standard import HGate, XGate, SdgGate, SGate, ZGate

from qiskit.aqua import QuantumInstance

from qiskit.aqua.operators.gradients import GradientBase



class QFI(GradientBase):
    """Compute the Quantum Fisher Information given a pure, parametrized quantum state."""

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the quantum Fisher information is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit=circuit, observable=None, quantum_instance=quantum_instance)

        warnings.simplefilter("once")



    def compute_qfi(self, parameters: Parameter, parameter_values: Dict,
                    qfi_circuits: Optional[Tuple[List[QuantumCircuit], List[QuantumCircuit]]] = None) -> np.ndarray:
        """Compute the entry of quantum Fisher Information with respect to the provided parameters.

        Args:
            parameters: The parameters with respect to which the quantum Fisher Information is computed.
            parameter_values: The values of the parameters with respect to which the quantum Fisher Information
            is computed.
            qfi_circuits: Tuple of lists of quantum circuits needed to compute the quantum Fisher Information.

        Returns: quantum Fisher Information
        """

        def get_exp_value(circuit: List[QuantumCircuit], operator: OperatorBase = Z) -> List[float]:
            r"""
            Evaluate the expectation value $\langle Z \rangle$ w.r.t. the ancilla qubit (named 'ancilla')
            Args:
                circuit: list of quantum circuits with a single qubit QuantumRegister with name 'ancilla'
                operator: Operator to get the correct expectation value.

            Returns: expectation value $\langle Z \rangle$ w.r.t. the 'ancilla' QuantumRegister

            """

            def prepare(qc: QuantumCircuit) -> Tuple[PauliOp, List[Qubit]]:
                r"""
                Get the operator and the corresponding QuantumRegister to evaluate $\langle Z \rangle$
                Args:
                    qc: This circuit prepares the state for which we want to evaluate $\langle Z
                    \rangle$ for the QuantumRegister named 'q'


                Returns:
                    Operator used for the expectation value evaluation and the corresponding qubit registers

                """
                evaluation_qubit = find_regs_by_name(qc, 'ancilla')
                qregs_list = circuit_item.qregs
                index_evaluation_qubit = qregs_list.index(evaluation_qubit)
                for i, qreg in enumerate(qregs_list):
                    if i == index_evaluation_qubit:
                        if i == 0:
                            qubit_op = operator
                        else:
                            qubit_op = operator ^ qubit_op
                    else:
                        if i == 0:
                            qubit_op = I ^ qreg.size
                        else:
                            qubit_op = I ^ qreg.size ^ qubit_op

                return qubit_op

            if not isinstance(circuit, list):
                circuit = [circuit]
            # TODO update CircuitSampler to facilitate circuit batching
            exp_vals = []
            for k, circuit_item in enumerate(circuit):
                # Transpile & assign parameter values
                circuit_item = transpile(circuit_item, backend=self._quantum_instance.backend)
                new_dict = {param: value for param, value in master_dict.items() if
                            param in circuit_item.parameters}
                circuit_item = circuit_item.assign_parameters(new_dict)
                # Construct circuits to evaluate the expectation values
                qubit_op = prepare(circuit_item)
                meas = ~StateFn(qubit_op)
                expect_op = meas @ StateFn(circuit_item)
                # Here, convert to Pauli measurement
                expect_op = PauliExpectation().convert(expect_op)
                exp_val = CircuitSampler(self._quantum_instance).convert(expect_op)
                exp_val = exp_val.eval()
                exp_vals.append(exp_val)

            return exp_vals

        master_dict = parameter_values

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        qfi_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        qfi_gates = {}
        # Loop throuh the parameters in the circuit
        params = []
        for param, elements in self._circuit._parameter_table.items():
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
        if qfi_circuits is None:
            qfi_circuits, qfi_phase_fix_circuits = self.construct_circuits(gates_to_parameters, qfi_coeffs, qfi_gates)
        else:
            qfi_circuits, qfi_phase_fix_circuits = qfi_circuits
        if len(qfi_circuits) > 0:
            qfi_exp_values = get_exp_value(qfi_circuits)
        if len(qfi_phase_fix_circuits) > 0:
            phase_fix_op = (X + 1j*Y) # see https://arxiv.org/pdf/quant-ph/0108146.pdf
            qfi_phase_fix_exp_values = get_exp_value(qfi_phase_fix_circuits, operator=phase_fix_op)

        phase_fix_values = np.zeros(len(gates_to_parameters), dtype=complex)
        counter_phase_fix = 0
        counter = 0
        # weighted sum of the circuit expectation values w.r.t. the coefficients
        for i in range(len(params)):
            for coeffs in qfi_coeffs[params[i]]:
                for coeff in coeffs:
                    phase_fix_values[i] += coeff * qfi_phase_fix_exp_values[counter_phase_fix]
                    counter_phase_fix += 1
            j = 0
            while j <= i:
                qfi[i, j] -= np.real(np.conj(phase_fix_values[i]) * phase_fix_values[j]) # Check
                for coeffs_i in qfi_coeffs[params[i]]:
                    for coeff_i in coeffs_i:
                        for coeffs_j in qfi_coeffs[params[j]]:
                            for coeff_j in coeffs_j:
                                # Quantum circuit already considers sign and if complex.
                                qfi[i, j] += np.abs(coeff_i) * np.abs(coeff_j) * qfi_exp_values[counter]
                                counter += 1
                        qfi[j, i] = qfi[i, j]
                        j += 1
        # Add correct pre-factor and return
        return 4*qfi

    def construct_circuits(self, parameterized_gates: Dict[Parameter, List[Gate]],
                           qfi_coeffs: Dict[Parameter, List[List[complex]]],
                           qfi_gates: Dict[Parameter, List[List[Instruction]]]) -> \
                           Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Generate the quantum Fisher Information circuits.

        Args:
            parameterized_gates: The dictionary of parameters and gates with respect to which the quantum Fisher
            Information is computed.
            qfi_coeffs: The values needed to compute the quantum Fisher Information for the parameterized gates.
                    For each parameter, the dict holds a list of all coeffs for all gates which are parameterized by
                    the parameter. {param:[[coeffs0],...]}
            qfi_gates: The gates needed to compute the quantum Fisher Information for the parameterized gates.
                    For each parameter, the dict holds a list of all gates to insert for all gates which are
                    parameterized by the parameter. {param:[[gates_to_insert0],...]}

        Returns:
            Two lists of quantum circuits which are needed to compute the quantum Fisher Information.
            If a parameter appears multiple times, one circuit is created per parameterized gates to be able to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """

        circuits = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        additional_qubits = ([ancilla], [])
        # create a copy of the original circuit with an additional ancilla register
        circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
        circuit.data = self._circuit.data
        params = list(parameterized_gates.keys())
        # apply Hadamard on ancilla
        self.insert_gate(circuit, parameterized_gates[params[0]][0], HGate(),
                    qubits=[ancilla])
        # Get the circuits needed to compute A_ij
        for i in range(len(params)): #loop over parameters
            j = 0
            while j <= i: #loop over parameters

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
                                        self.insert_gate(qfi_circuit, parameterized_gates[params[0]][0], SdgGate(),
                                                    qubits=[ancilla])
                                    else:
                                        self.insert_gate(qfi_circuit, parameterized_gates[params[0]][0], ZGate(),
                                                    qubits=[ancilla])
                                else:
                                    if complex:
                                        self.insert_gate(qfi_circuit, parameterized_gates[params[0]][0], SGate(),
                                                    qubits=[ancilla])

                                self.insert_gate(qfi_circuit, parameterized_gates[params[0]][0], XGate(),
                                            qubits=[ancilla])

                                # Insert controlled, intercepting gate - controlled by |1>
                                self.insert_gate(qfi_circuit, parameterized_gates[params[i]][m], gate_to_insert_i,
                                                                         additional_qubits=additional_qubits)

                                self.insert_gate(qfi_circuit, gate_to_insert_i, XGate(), qubits=[ancilla], after=True)

                                # Insert controlled, intercepting gate - controlled by |0>
                                self.insert_gate(qfi_circuit, parameterized_gates[params[j]][n], gate_to_insert_j,
                                                                         additional_qubits=additional_qubits)

                                # Remove redundant gates
                                qfi_circuit = self.trim_circuit(qfi_circuit, parameterized_gates[params[i]][m])

                                qfi_circuit.h(ancilla)
                                circuits += [qfi_circuit]
                j += 1
        circuits_phase_fix = []
        for i in range(len(params)):  # loop over parameters

                # construct the phase fix circuits
                for m, gates_to_insert_i in enumerate(qfi_gates[params[i]]):
                    for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                        # create a copy of the original circuit with the same registers
                        qfi_circuit = QuantumCircuit(*circuit.qregs)
                        qfi_circuit.data = circuit.data
                        # Insert controlled, intercepting gate
                        self.insert_gate(qfi_circuit, parameterized_gates[params[i]][m],
                                                                 gate_to_insert_i,
                                                                 additional_qubits=additional_qubits)

                        qfi_circuit = self.trim_circuit(qfi_circuit, parameterized_gates[params[i]][m])


                        circuits_phase_fix += [qfi_circuit]
        return circuits, circuits_phase_fix
