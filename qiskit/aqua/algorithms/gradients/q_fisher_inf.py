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

"""The module for Quantum Natural Gradients."""

from typing import Optional, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate, Qubit
from qiskit.extensions.standard import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, \
    CZGate, U1Gate, U2Gate, U3Gate, RXXGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate

from qiskit.aqua import QuantumInstance, AquaError

from .gradient import Gradient


class QuantumFisherInf(Gradient):
    """Compute the quantum Fisher Information given a pure, parametrized quantum state."""

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the quantum Fisher information is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit, quantum_instance)

    def compute_qfi(self, parameters: Parameter) -> float:
        """Compute the entry of quantum Fisher Information with respect to the provided parameters.

        Args:
            parameters: The parameters with respect to which the quantum Fisher Information is computed.
        """
        # TODO compute gradient based on the gradient circuits
        qfi_circuits = self.construct_circuits(parameters)

        # TODO add ancilla to compensate for the potential phase mismatch

        # TODO delete this below here
        print('The current circuit is')
        print(self._circuit.draw())
        print('We are computing the derivative with respect to', parameters)
        print('The gradient circuits are:')
        for circuit in qfi_circuits:
            print(circuit.draw())

    def construct_circuits(self, parameters: Parameter) -> List[QuantumCircuit]:
        """Generate the quantum Fisher Information circuits.

        Args:
            parameters: The parameters with respect to which the quantum Fisher Information is computed.

        Returns:
            A list of the circuits with the inserted gates. If a parameter appears multiple times,
            one circuit is created per parameterized gates to be able to compute the
            product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        parameterized_gates = [element[0] for element in self._circuit._parameter_table[parameters]]

        circuits = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        for reference_gate in parameterized_gates:
            # get the controlled gate (raises an error if the parameterized gate is not supported)
            entangler_gate = QuantumFisherInf.get_controlled_gate(reference_gate)

            # create a copy of the original circuit with the same registers
            gradient_circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
            gradient_circuit.data = self._circuit.data

            # TODO add pre-operations on the ancilla qubit a

            additional_qubits = ([ancilla], [])
            success = QuantumFisherInf.insert_gate(gradient_circuit, reference_gate, entangler_gate,
                                                  additional_qubits=additional_qubits)

            # TODO trim circuit

            # TODO add post-operation on the ancilla qubit

            if not success:
                raise AquaError('Could not insert the controlled gate, something went wrong!')
            circuits += [gradient_circuit]

        return circuits

    @staticmethod
    def insert_gate(circuit: QuantumCircuit,
                    reference_gate: Gate,
                    gate_to_insert: Gate,
                    qubits: Optional[List[Qubit]] = None,
                    additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None) -> bool:
        """Insert a gate into the circuit.

        Args:
            circuit: The circuit onto which the gare is added.
            reference_gate: A gate instance before or after which a gate is inserted.
            gate_to_insert: The gate to be inserted.
            qubits: The qubits on which the gate is inserted. If None, the qubits of the
                reference_gate are used.
            additional_qubits: If qubits is None and the qubits of the reference_gate are
                used, this can be used to specify additional qubits before (first list in
                tuple) or after (second list in tuple) the qubits.

        Returns:
            True, if the insertion has been successful, False otherwise.
        """
        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                qubits = qubits or op[1]
                if additional_qubits:
                    qubits = additional_qubits[0] + qubits + additional_qubits[1]
                op_to_insert = (gate_to_insert, qubits, [])
                insertion_index = i
                circuit.data.insert(insertion_index, op_to_insert)
                return True

        return False

    @staticmethod
    def get_controlled_gate(gate: Gate) -> Tuple[List[complex], List[Gate]]:
        """Get the controlled gate for the natural gradient.

        Currently, only pauli rotation gates are suppported.

        Args:
            gate: The gate for which the derivative is being computed.

        Returns:
            The coefficients and the gates used for the metric computation.

        Raises:
            TypeError: If the input gate is not a supported parametrized gate.
        """
        if isinstance(gate, RXGate):
            return [CXGate()]
        if isinstance(gate, RYGate):
            return [CYGate()]
        if isinstance(gate, RZGate) or isinstance(gate, U1Gate):
            return CZGate()

        raise TypeError('Unrecognized parametrized gate, {}'.format(gate))
