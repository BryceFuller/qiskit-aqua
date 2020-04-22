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

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile

from qiskit.circuit import Parameter, Gate, ControlledGate, Qubit
from qiskit.extensions.standard import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, CZGate,\
    U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, XGate, \
    SdgGate, SGate, ZGate

from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.aqua.utils.run_circuits import find_regs_by_name

from qiskit.aqua import QuantumInstance, AquaError

from .gradient import Gradient


class QuantumFisherInf(Gradient):
    """Compute the quantum Fisher Information given a pure, parametrized quantum state."""
    # TODO extend to mixed states

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the quantum Fisher information is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit, quantum_instance)
        print('I may have done a hacky thing with the construct_evaluation_circuit method of the '
              'weighted pauli operator. I.e. construct circuit did not work with lists of quantum '
              'registers but only full registers. Wait for refactoring of operator.')

    def compute_qfi(self, parameters: Parameter, parameter_values: List,
                    qfi_circuits: Optional[Tuple[List[QuantumCircuit], List[QuantumCircuit]]] = None) -> np.ndarray:
        """Compute the entry of quantum Fisher Information with respect to the provided parameters.

        Args:
            parameters: The parameters with respect to which the quantum Fisher Information is computed.
            parameter_values: The values of the parameters with respect to which the quantum Fisher Information
            is computed.
            qfi_circuits: Tuple of lists of quantum circuits needed to compute the quantum Fisher Information.

        Returns: quantum Fisher Information
        """

        def get_exp_value(circuit: List[QuantumCircuit]) -> List[float]:
            r"""
            Evaluate the expectation value $\langle Z \rangle$ w.r.t. the ancilla qubit (named 'ancilla')
            Args:
                circuit: list of quantum circuits with a single qubit QuantumRegister with name 'ancilla'

            Returns: expectation value $\langle Z \rangle$ w.r.t. the 'ancilla' QuantumRegister

            """

            def prepare(qc: QuantumCircuit, single_qubit_mem: bool = False) -> \
                    Tuple[WeightedPauliOperator, List[Qubit]]:
                r"""
                Get the operator and the corresponding QuantumRegister to evaluate $\langle Z \rangle$
                Args:
                    qc: This circuit prepares the state for which we want to evaluate $\langle Z
                    \rangle$ for the QuantumRegister named 'q'
                    single_qubit_mem: Boolean to denote whether we want to evaluate only the measurement of the target
                    qubit register or all registers (for measurement error mitigation)

                Returns:
                    Operator used for the expectation value evaluation and the corresponding qubit registers

                """
                evaluation_qubit = find_regs_by_name(qc, 'ancilla')
                if single_qubit_mem:
                    pauli = 'Z'
                    qregs_list = evaluation_qubit
                else:
                    qregs_list = circuit_item.qregs
                    index_evaluation_qubit = qregs_list.index(evaluation_qubit)
                    pauli = ''
                    for i, qreg in enumerate(qregs_list):
                        if i == index_evaluation_qubit:
                            pauli = 'Z' + pauli
                        else:
                            pauli = 'I' * qreg.size + pauli

                    temp = []
                    for element in qregs_list:
                        for i in range(element.size):
                            temp.append(element[i])
                    qregs_list = temp

                qubit_op = WeightedPauliOperator([[1, Pauli.from_label(pauli)]])
                return qubit_op, qregs_list

            # TODO problem with instruction -> decompose() not optimal RZ -> U1 wrong derivative

            sv_mode = False
            if self._quantum_instance.is_statevector:
                sv_mode = True
            if not isinstance(circuit, list):
                circuit = [circuit]
            qc = []
            qubit_ops = []
            # TODO enable measurement error mitigation when circuitd in qobj use different physical qubits
            # Construct master dictionary for parameter assignment
            master_dict = {}
            for q, param in enumerate(self._circuit.parameters):
                master_dict[param] = parameter_values[q]
            for k, circuit_item in enumerate(circuit):
                # Transpile & assign parameter values
                circuit_item = transpile(circuit_item, backend=self._quantum_instance.backend)
                new_dict = {param: value for param, value in master_dict.items() if
                            param in circuit_item.parameters}
                circuit_item = circuit_item.assign_parameters(new_dict)
                # Construct circuits to evaluate the expectation values
                qubit_op, qregs_list = prepare(circuit_item, single_qubit_mem=False)
                qc.extend(qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode, wave_function=circuit_item,
                                                          qr=qregs_list, circuit_name_prefix='circuits' + str(k)))
                qubit_ops.append(qubit_op)

            success = False
            counter = 0

            while not success:
                # This prevents errors if a hardware call may return an error.
                try:
                    result = self._quantum_instance.execute(qc)
                    success = True
                except Exception:
                    counter += 1
                if counter > 10:
                    raise AquaError('Get expectation value failed.')
                    break

            avg = []
            for k, circuit_item in enumerate(circuit):
                avg_temp, _ = qubit_ops[k].evaluate_with_result(statevector_mode=sv_mode, result=result,
                                                                circuit_name_prefix='circuits' + str(k))
                avg.append(avg_temp)
            return avg

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)

        parameterized_gates = []
        for param, elements in self._circuit._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        qfi_coeffs = []
        qfi_gates = []
        for reference_gate in parameterized_gates:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs, gates = QuantumFisherInf.get_coeffs_gates(reference_gate)
                qfi_coeffs.append(coeffs)
                qfi_gates.append(gates)
        if qfi_circuits is None:
            qfi_circuits, qfi_phase_fix_circuits = self.construct_circuits(parameterized_gates)
        else:
            qfi_circuits, qfi_phase_fix_circuits = qfi_circuits
        if len(qfi_circuits) > 0:
            qfi_exp_values = get_exp_value(qfi_circuits)
        if len(qfi_phase_fix_circuits) > 0:
            qfi_phase_fix_exp_values = get_exp_value(qfi_phase_fix_circuits)

        phase_fix_values = np.zeros(len(parameterized_gates), dtype=complex)
        counter_phase_fix = 0
        counter = 0
        for i in range(len(parameterized_gates)):
            for k, coeff in enumerate(qfi_coeffs[i]):
                # print('coeff ', coeff)
                # print('phase fix value', qfi_phase_fix_exp_values[counter_phase_fix])
                # print('qfi ', qfi)
                # print(coeff * qfi_phase_fix_exp_values[counter_phase_fix])
                phase_fix_values[i] += coeff * qfi_phase_fix_exp_values[counter_phase_fix]
                counter_phase_fix += 1
            # print(phase_fix_values)
            j = 0
            while j <= i:
                qfi[i, j] -= np.real(np.conj(phase_fix_values[i]) * phase_fix_values[j])
                # print('qfi ', qfi)
                for coeff_i in qfi_coeffs[i]:
                    for coeff_j in qfi_coeffs[j]:
                        qfi[i, j] += np.abs(coeff_i) * np.abs(coeff_j) * qfi_exp_values[counter]
                        counter += 1
                qfi[j, i] = qfi[i, j]
                j += 1
                # print('qfi ', qfi)

        # TODO delete below
        # for circuit in qfi_phase_fix_circuits:
            # print(circuit)
        # print(phase_fix_values)
        # print(qfi_coeffs)
        # print(qfi_phase_fix_exp_values)

        # Add correct pre-factor and return
        return 4*qfi

    def construct_circuits(self, parameterized_gates: List[Gate]) -> \
            Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Generate the quantum Fisher Information circuits.

        Args:
            parameterized gates: The list of parameters and gates with respect to which the quantum Fisher Information
            is computed.
            parameter_values: The values of the parameters with respect to which the quantum Fisher Information
            is computed.

        Returns:
            Two lists of quantum circuits which are needed to compute the quantum Fisher Information.
            If a parameter appears multiple times, one circuit is created per parameterized gates to be able to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        qfi_coeffs = []
        qfi_gates = []
        for reference_gate in parameterized_gates:
                # get the coefficients and controlled gates
                coeffs, gates = QuantumFisherInf.get_coeffs_gates(reference_gate)
                qfi_coeffs.append(coeffs)
                qfi_gates.append(gates)

        circuits = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        additional_qubits = ([ancilla], [])

        for i in range(len(parameterized_gates)):
            j = 0
            while j <= i:
                # create a copy of the original circuit with the same registers
                circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
                circuit.data = self._circuit.data
                success = QuantumFisherInf.insert_gate(circuit, parameterized_gates[0], HGate(),
                                                       qubits=[ancilla])
                if not success:
                    raise AquaError('Could not insert the controlled gate, something went wrong!')

                # construct the circuits

                for k, gate_to_insert_i in enumerate(qfi_gates[i]):
                    for l, gate_to_insert_j in enumerate(qfi_gates[j]):
                        qfi_circuit = QuantumCircuit(*circuit.qregs)
                        qfi_circuit.data = circuit.data
                        # Fix phase ancilla
                        phase = np.sign(np.conj(qfi_coeffs[i][k]))*np.sign(qfi_coeffs[j][l])
                        if np.iscomplex(qfi_coeffs[i][k]):
                            if np.iscomplex(qfi_coeffs[j][l]):
                                phase *= (-1)
                            else:
                                phase *= 1j
                        else:
                            if np.iscomplex(qfi_coeffs[j][l]):
                                phase *= 1j
                        if phase == 1j:
                            success = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[0], SGate(),
                                                                   qubits=[ancilla])
                            if not success:
                                raise AquaError('Could not insert the controlled gate, something went wrong!')
                        elif phase == -1j:
                            success = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[0], SdgGate(),
                                                                   qubits=[ancilla])
                            if not success:
                                raise AquaError('Could not insert the controlled gate, something went wrong!')
                        elif phase == -1:
                            success = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[0], ZGate(),
                                                                   qubits=[ancilla])
                            if not success:
                                raise AquaError('Could not insert the controlled gate, something went wrong!')

                        success = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[0], XGate(),
                                                               qubits=[ancilla])
                        if not success:
                            raise AquaError('Could not insert the controlled gate, something went wrong!')
                        # (gate0, gate1) -> gate0[q0], gate1[q1]
                        if isinstance(gate_to_insert_i, tuple):
                            success_i = True
                            for op in circuit.data:
                                if op[0] == parameterized_gates[i]:
                                    qubits = op[1]
                            for p, qubit in enumerate(qubits):
                                success_i &= QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[i],
                                                                         gate_to_insert_i[p], qubits=[qubit],
                                                                         additional_qubits=additional_qubits)
                        else:
                            success_i = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[i],
                                                                     gate_to_insert_i,
                                                                     additional_qubits=additional_qubits)

                        if not success_i:
                            raise AquaError('Could not insert the controlled gate, something went wrong!')

                        success = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[i], XGate(),
                                                               qubits=[ancilla])
                        if not success:
                            raise AquaError('Could not insert the controlled gate, something went wrong!')

                        # (gate0, gate1) -> gate0[q0], gate1[q1]
                        if isinstance(gate_to_insert_j, tuple):
                            success_j = True
                            for op in circuit.data:
                                if op[0] == parameterized_gates[j]:
                                    qubits = op[1]
                            for p, qubit in enumerate(qubits):
                                success_j &= QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[j],
                                                                         gate_to_insert_j[p], qubits=[qubit],
                                                                         additional_qubits=additional_qubits)
                        else:
                            success_j = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[j],
                                                                     gate_to_insert_j,
                                                                     additional_qubits=additional_qubits)
                        if not success_j:
                            raise AquaError('Could not insert the controlled gate, something went wrong!')

                        # Remove redundant gates
                        qfi_circuit = QuantumFisherInf.trim_circuit(qfi_circuit, parameterized_gates[i])

                        qfi_circuit.h(ancilla)
                        circuits += [qfi_circuit]
                j += 1

        circuits_phase_fix = []
        for i in range(len(parameterized_gates)):
                # create a copy of the original circuit with the same registers
                circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
                circuit.data = self._circuit.data
                success = QuantumFisherInf.insert_gate(circuit, parameterized_gates[0], HGate(),
                                                               qubits=[ancilla])
                if not success:
                    raise AquaError('Could not insert the controlled gate, something went wrong!')

                # construct the phase fix circuits

                for k, gate_to_insert_i in enumerate(qfi_gates[i]):
                    qfi_circuit = QuantumCircuit(*circuit.qregs)
                    qfi_circuit.data = circuit.data

                    # (gate0, gate1) -> gate0[q0], gate1[q1]
                    if isinstance(gate_to_insert_i, tuple):
                        success_i = True
                        for op in circuit.data:
                            if op[0] == parameterized_gates[i]:
                                qubits = op[1]
                        for p, qubit in enumerate(qubits):
                            success_i &= QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[i],
                                                                      gate_to_insert_i[p], qubits=[qubit],
                                                                      additional_qubits=additional_qubits)
                    else:
                        success_i = QuantumFisherInf.insert_gate(qfi_circuit, parameterized_gates[i],
                                                                 gate_to_insert_i,
                                                                 additional_qubits=additional_qubits)
                    if not success_i:
                        raise AquaError('Could not insert the controlled gate, something went wrong!')
                    # (gate0, gate1) -> gate0[q0], gate1[q1]

                        # Remove redundant gates
                    qfi_circuit = QuantumFisherInf.trim_circuit(qfi_circuit, parameterized_gates[i])

                    qfi_circuit.h(ancilla)
                    circuits_phase_fix += [qfi_circuit]

        return circuits, circuits_phase_fix

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
        if isinstance(gate_to_insert, IGate):
            return True
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

    # Not needed
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

    @staticmethod
    def get_coeffs_gates(gate: Gate) -> Tuple[List[complex], List[Gate]]:
        """Get the ancilla-controlled gates for the quantum Fisher Information.

        Notably, if gate is a two-qubit gate acting on qubits q0, q1 and the returned gate list consists of a gate
        tuple (gate0, gate1) then gate0 must act on q0 and gate1 on q1.

        Currently, not all parametrized gates are supported.

        Args:
            gate: The gate for which the derivative is being computed.

        Returns:
            The coefficients and the gates used for the metric computation for each parameter of the respective gates.

        Raises:
            TypeError: If the input gate is not a supported parametrized gate.
        """

        if isinstance(gate, U1Gate):
            # theta
            return [0.5j, -0.5j], [IGate(), CZGate()]
        # TODO Extend to gates with multiple parameters
        # if isinstance(gate, U2Gate):
        #     # TODO Think a little longer how we can reformulte the derivative suitably. - Commutation Relations
        #     # theta, phi
        #     return [[0.5j], [-0.5j]], [[?], [CZGate]]
        # if isinstance(gate, U3Gate):
        #     # TODO Think a little longer how we can reformulte the derivative suitably. - Commutation Relations
        #     # theta, lambda, phi
        #     return [[0.5j], [-0.5j], [-0.5j]], [[?], [?], [CZGate]]
        if isinstance(gate, RXGate):
            # theta
            return [-0.5j], [CXGate()]
        if isinstance(gate, RYGate):
            # theta
            return [-0.5j], [CYGate()]
        if isinstance(gate, RZGate):
            # theta
            # Note that the implemented RZ gate is not an actual RZ gate but [[1, 0], [0, e^i\theta]]
            return [-0.5j], [CZGate()]
        if isinstance(gate, RXXGate):
            # theta
            return [-0.5j], [CXGate()]
        if isinstance(gate, RYYGate):
            # theta
            return [-0.5j], [CYGate()]
        if isinstance(gate, RZZGate):
            # theta
            return [-0.5j], [(CZGate(), CZGate())]
        # TODO wait until this gate is fixed
        # if isinstance(gate, RZXGate):
        #     # theta
        #     return [[-0.5j]], [[(CZGate, CXGate)]]
        if isinstance(gate, CRXGate):
            # theta
            return [-0.25j, +0.25j], [(IGate(), CXGate()), (CZGate(), CXGate())]
        if isinstance(gate, CRYGate):
            # theta
            return [-0.25j, +0.25j], [(IGate(), CYGate()), (CZGate(), CYGate())]
        if isinstance(gate, CRZGate):
            # theta
            # Note that the implemented RZ gate is not an actual RZ gate but [[1, 0], [0, e^i\theta]]
            return [-0.25j, +0.25j], [(IGate(), CZGate()), CZGate()]
        if isinstance(gate, CU1Gate):
            # theta
            return [0.25j, -0.25j, -0.25j, 0.25j], [IGate(), (IGate(), CZGate()), (CZGate(), IGate()),
                                                    (CZGate(), CZGate())]

        r'''
        TODO multi-controlled-$U(\theta)$ for $m$ controlls:
        $\frac{1}{2^m}\Bigotimes\limits_i=0^{m-1}(I\-Z)\otimes \frac{\partial U}{\partial\theta} $
        # for self.num_ctrl_qubits
        '''

        raise TypeError('Unrecognized parametrized gate, {}'.format(gate))

    @staticmethod
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
        for param, elements in circuit._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                trimmed_circuit = QuantumCircuit(*circuit.qregs)
                trimmed_circuit.data = circuit.data[:i]
                return trimmed_circuit

        raise AquaError('The reference gate is not in the given quantum circuit.')
