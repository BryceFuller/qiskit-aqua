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

"""Gradient utils"""

from typing import Optional, Tuple, List
from itertools import product

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, ControlledGate, Qubit, Instruction
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, U1Gate, \
                                       U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, IGate, ZGate

from qiskit.aqua import AquaError

#Todo move to gradient base class

def gate_gradient_dict(gate: Gate) -> List[Tuple[List[complex], List[Instruction]]]:

    """Given a parameterized gate U(theta) with derivative dU(theta)/dtheta = sum_ia_iU(theta)V_i.
       This function returns a:=[a_0, ...] and V=[V_0, ...]
       Suppose U takes multiple parameters, i.e., U(theta^0, ... theta^k).
       The returned coefficients and gates are ordered accordingly.
       Only parameterized Qiskit gates are supported.

       Args:
            gate: The gate for which the derivative is being computed.

       Returns:
            The coefficients and the gates used for the metric computation for each parameter of the respective gates.
            [([a^0], [V^0]) ..., ([a^k], [V^k])]

       Raises:
            TypeError: If the input gate is not a supported parametrized gate."""

    if isinstance(gate, U1Gate):
        # theta
        return [([0.5j, -0.5j], [IGate, CZGate])]
    if isinstance(gate, U2Gate): #Going to be deprecated
        # theta, phi
        return [([-0.5j], [CZGate]), ([0.5j], [CZGate])]
    if isinstance(gate, U3Gate):
        # theta, lambda, phi
        return [([-0.5j], [CZGate]), ([+0.5j], [CZGate]), ([-0.5j], [CZGate])]
    if isinstance(gate, RXGate):
        # theta
        return [([-0.5j], [CXGate()])]
    if isinstance(gate, RYGate):
        # theta
        return [([-0.5j], [CYGate()])]
    if isinstance(gate, RZGate):
        # theta
        return [([-0.5j], [CZGate()])]
    if isinstance(gate, RXXGate):
        # theta
        cxx_circ = QuantumCircuit(3)
        cxx_circ.cx(0, 1)
        cxx_circ.cx(0, 2)
        cxx = cxx_circ.to_instruction()
        return [([-0.5j], [cxx])]
    if isinstance(gate, RYYGate):
        # theta
        cyy_circ = QuantumCircuit(3)
        cyy_circ.cy(0, 1)
        cyy_circ.cy(0, 2)
        cyy = cyy_circ.to_instruction()
        return [([-0.5j], [cyy])]
    if isinstance(gate, RZZGate):
        # theta
        czz_circ = QuantumCircuit(3)
        czz_circ.cz(0, 1)
        czz_circ.cz(0, 2)
        czz = czz_circ.to_instruction()
        return [([-0.5j], [czz])]
    if isinstance(gate, RZXGate):
        # theta
        czx_circ = QuantumCircuit(3)
        czx_circ.cx(0, 1)
        czx_circ.cz(0, 2)
        czx = czx_circ.to_instruction()
        return [([-0.5j], [czx])]
    if isinstance(gate, ControlledGate):
        # TODO support arbitrary control states
        if gate.ctrl_state != 2**gate.num_ctrl_qubits - 1:
            raise AquaError('Function only support controlled gates with control state `1` on all control qubits.')
        base_coeffs_gates = gate_gradient_dict(gate.base_gate)
        coeffs_gates = []
        # The projectors needed for the gradient of a controlled gate are integrated by a sum of gates.
        # The following line generates the decomposition gates.
        proj_gates_controlled = [[(-1) ** p.count(ZGate()), p] for p in product([IGate(), ZGate()], repeat=gate.num_ctrl_qubits)]
        for base_coeffs, base_gates in base_coeffs_gates: # loop over parameters
            coeffs = [c / (2 ** gate.num_ctrl_qubits) for c in base_coeffs]
            gates = []
            for phase, proj_gates in proj_gates_controlled:
                base_coeffs.extend(phase * c for c in base_coeffs)
                for base_gate in base_gates:
                    controlled_circ = QuantumCircuit(gate.num_ctrl_qubits + gate.num_qubits + 1)
                    for i, proj_gate in enumerate(proj_gates):
                        if isinstance(proj_gate, ZGate):
                            controlled_circ.cz(0, i)
                    if not isinstance(base_gate, IGate):
                        controlled_circ.append(base_gate, [0, range(gate.num_ctrl_qubits + 1, gate.num_ctrl_qubits +
                                                                gate.num_qubits + 1)])
                    # TODO make sure that the list before does the right thing. i.e. append a controlled gate to the
                    # TODO ancilla 0 and the gates considering the base_gate
                    gates.append(controlled_circ.to_instruction())
            coeffs_gates.append((coeffs, gates))
        return coeffs_gates

    raise TypeError('Unrecognized parametrized gate, {}'.format(gate))


def insert_gate(circuit: QuantumCircuit,
                reference_gate: Gate,
                gate_to_insert: Instruction,
                qubits: Optional[List[Qubit]] = None,
                additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None,
                after: bool = False):

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
        after: If the gate_to_insert should be inserted after the reference_gate set True.
    """

    if isinstance(gate_to_insert, IGate):
        return
    else:
        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                qubits = qubits or op[1]
                if additional_qubits:
                    qubits = additional_qubits[0] + qubits + additional_qubits[1]
                op_to_insert = (gate_to_insert, qubits, [])
                if after:
                    insertion_index = i+1
                else:
                    insertion_index = i
                circuit.data.insert(insertion_index, op_to_insert)
                return
        raise AquaError('Could not insert the controlled gate, something went wrong!')


def trim_circuit(circuit: QuantumCircuit, reference_gate: Gate) -> QuantumCircuit:
    """Trim the given quantum circuit before the reference gate.

    Args:
        circuit: The circuit onto which the gare is added.
        reference_gate: A gate instance before or after which a gate is inserted.

    Returns:
        The trimmed circuit.

    Raise:
        AquaError: If the reference gate is not part of the given circuit.
    """
    parameterized_gates = []
    for _, elements in circuit._parameter_table.items():
        for element in elements:
            parameterized_gates.append(element[0])

    for i, op in enumerate(circuit.data):
        if op[0] == reference_gate:
            trimmed_circuit = QuantumCircuit(*circuit.qregs)
            trimmed_circuit.data = circuit.data[:i]
            return trimmed_circuit

    raise AquaError('The reference gate is not in the given quantum circuit.')

# For now not needed
# @staticmethod
# def replace_gate(circuit: QuantumCircuit,
#                 gate_to_replace: Gate,
#                 gate_to_insert: Gate,
#                 qubits: Optional[List[Qubit]] = None,
#                 additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None) -> bool:
#     """Insert a gate into the circuit.
#
#     Args:
#         circuit: The circuit onto which the gare is added.
#         gate_to_replace: A gate instance which shall be replaced.
#         gate_to_insert: The gate to be inserted instead.
#         qubits: The qubits on which the gate is inserted. If None, the qubits of the
#             reference_gate are used.
#         additional_qubits: If qubits is None and the qubits of the reference_gate are
#             used, this can be used to specify additional qubits before (first list in
#             tuple) or after (second list in tuple) the qubits.
#
#     Returns:
#         True, if the insertion has been successful, False otherwise.
#     """
#     for i, op in enumerate(circuit.data):
#         if op[0] == gate_to_replace:
#             circuit.data = circuit.data.pop(i) # remove gate
#             if isinstance(gate_to_insert, IGate()):
#                 return True
#             #TODO check qubits placing
#             qubits = qubits or op[1][-(gate_to_replace.num_qubits - gate_to_replace.num_clbits):]
#             if additional_qubits:
#                 qubits = additional_qubits[0] + qubits + additional_qubits[1]
#             op_to_insert = (gate_to_insert, qubits, [])
#             insertion_index = i
#             circuit.data.insert(insertion_index, op_to_insert)
#             return True
#
#     return False

