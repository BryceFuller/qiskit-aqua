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
from qiskit.extensions.standard import RXGate, RYGate, RZGate, CXGate, CYGate, CZGate

from qiskit.aqua import QuantumInstance, AquaError

from .gradient import Gradient


class NaturalGradient(Gradient):
    """Compute the natural gradient of a quantum circuit."""

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the gradient is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit, quantum_instance)

    def compute_gradient(self, parameter: Parameter) -> float:
        """Compute the gradient with respect to the provided parameter.

        Args:
            parameter: The parameter with respect to which the gradient is computed.
        """
        # TODO
        print('The current circuit is')
        print(self._circuit.draw())
        print('We are computing the derivative with respect to', parameter)
        print('The gradient circuits are:')
        for circuit in self.construct_circuits(parameter):
            print(circuit.draw())

    def construct_circuits(self, parameter: Parameter, before: bool = True) -> List[QuantumCircuit]:
        """Generate the gradient circuits.

        Args:
            parameter: The parameter with respect to which the gradient is computed.
            before: Whether the controlled gates are applied before or after the parameterized
                gates.

        Returns:
            A list of the circuits with the inserted gates. If a parameter appears multiple times,
            one circuit is created per parameterized gates to be able to compute the
            product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        parameterized_gates = [element[0] for element in self._circuit._parameter_table[parameter]]

        circuits = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        for reference_gate in parameterized_gates:
            # get the entangler gate (raises an error if the parameterized gate is not supported)
            entangler_gate = NaturalGradient.get_entangler_gate(reference_gate)

            # create a copy of the original circuit with the same registers
            gradient_circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
            gradient_circuit.data = self._circuit.data

            additional_qubits = ([ancilla], [])
            success = NaturalGradient.insert_gate(gradient_circuit, reference_gate, entangler_gate,
                                                  additional_qubits=additional_qubits,
                                                  before=before)

            if not success:
                raise AquaError('Could not insert the controlled gate, something went wrong!')
            circuits += [gradient_circuit]

        return circuits

    @staticmethod
    def insert_gate(circuit: QuantumCircuit,
                    reference_gate: Gate,
                    gate_to_insert: Gate,
                    qubits: Optional[List[Qubit]] = None,
                    additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None,
                    before: bool = True) -> bool:
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
            before: If True insert the gate before the reference, else after.

        Returns:
            True, if the insertion has been successful, False otherwise.
        """
        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                qubits = qubits or op[1]
                if additional_qubits:
                    qubits = additional_qubits[0] + qubits + additional_qubits[1]
                op_to_insert = (gate_to_insert, qubits, [])
                insertion_index = i + int(before is False)
                circuit.data.insert(insertion_index, op_to_insert)
                return True

        return False

    @staticmethod
    def get_entangler_gate(gate: Gate) -> ControlledGate:
        """Get the entangler gate for the natural gradient.

        Currently, only pauli rotation gates are suppported.

        Args:
            gate: The gate for which the derivative is being computed.

        Returns:
            The controlled gate used for derivative computation.

        Raises:
            TypeError: If the input gate is not a Pauli rotation gate.
        """
        if isinstance(gate, RXGate):
            return CXGate()
        if isinstance(gate, RYGate):
            return CYGate()
        if isinstance(gate, RZGate):
            return CZGate()

        raise TypeError('Unrecognized Pauli rotation gate, {}'.format(gate))
