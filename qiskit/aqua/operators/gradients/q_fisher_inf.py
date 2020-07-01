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

from typing import Optional, Tuple, List, Dict
import warnings
import sys

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile

from qiskit.circuit import Parameter, Gate, ControlledGate, Qubit, Instruction

from qiskit.aqua.operators import PauliOp, X, Y, Z, I, CircuitSampler, StateFn, OperatorBase, PauliExpectation, ListOp
from qiskit.quantum_info import Pauli
from qiskit.aqua.utils.run_circuits import find_regs_by_name

from qiskit.extensions.standard import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, CZGate,\
    U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, XGate, \
    SdgGate, SGate, ZGate

from qiskit.aqua import QuantumInstance, AquaError

from .gradient import Gradient
from .gradients_utils import gate_gradient_dict, insert_gate, trim_circuit


class QuantumFisherInf(Gradient):
    """Compute the quantum Fisher Information given a pure, parametrized quantum state."""
    # TODO product rule

    # TODO extend to mixed states

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the quantum Fisher information is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit, quantum_instance)

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

                temp = []
                for element in qregs_list:
                    for i in range(element.size):
                        temp.append(element[i])
                qregs_list = temp

                return qubit_op, qregs_list

            # sv_mode = False
            # if self._quantum_instance.is_statevector:
            #     sv_mode = True
            if not isinstance(circuit, list):
                circuit = [circuit]
            # TODO update CircuitSampler to facilitate circuit batching
            exp_vals = []
            # test = []
            for k, circuit_item in enumerate(circuit):
                # Transpile & assign parameter values
                circuit_item = transpile(circuit_item, backend=self._quantum_instance.backend)
                new_dict = {param: value for param, value in master_dict.items() if
                            param in circuit_item.parameters}
                circuit_item = circuit_item.assign_parameters(new_dict)
                # Construct circuits to evaluate the expectation values
                qubit_op, qregs_list = prepare(circuit_item)
                meas = ~StateFn(qubit_op)  # ~ is the same as .adjoint()
                expect_op = meas @ StateFn(circuit_item)
                # I can already call eval() on expect_op, but it will do the evaluation by matrix multiplication.
                # Here, convert to Pauli measurement
                expect_op = PauliExpectation().convert(expect_op)
                # test.append(expect_op)
                exp_val = CircuitSampler(self._quantum_instance).convert(expect_op)
                exp_val = exp_val.eval()
                exp_vals.append(exp_val)

                # executed_op = CircuitSampler(self._quantum_instance).convert(expect_op)
                # exp_vals.append(executed_op.eval())
            # test = ListOp(test)
            # exp_val = CircuitSampler(self._quantum_instance).convert(test)
            # exp_vals = exp_val.eval()
            return exp_vals

                # # TODO Do we still need qregs_list? How to get the exp value? - Circuit Sampler?
                # qc.extend(qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode, wave_function=circuit_item,
                #                                           qr=qregs_list, circuit_name_prefix='circuits' + str(k)))
                # qubit_ops.append(qubit_op)

            # success = False
            # counter = 0
            #
            # while not success:
            #     # This prevents errors if a hardware call may return an error.
            #     try:
            #         result = self._quantum_instance.execute(qc)
            #         success = True
            #     except Exception:
            #         counter += 1
            #     if counter > 10:
            #         raise AquaError('Get expectation value failed.')
            #         break
            #
            # avg = []
            # for k, circuit_item in enumerate(circuit):
            #     avg_temp, _ = qubit_ops[k].evaluate_with_result(statevector_mode=sv_mode, result=result,
            #                                                     circuit_name_prefix='circuits' + str(k))
            #     avg.append(avg_temp)
            # return avg

        master_dict = parameter_values

        qfi = np.zeros((len(parameters), len(parameters)), dtype=complex)

        # here dictionary w.r.t. params
        #-----------------
        # # List with all parameterized gates
        # parameterized_gates = []
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
                coeffs_gates = gate_gradient_dict(element[0])
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
        insert_gate(circuit, parameterized_gates[params[0]][0], HGate(),
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
                                qfi_circuit = QuantumCircuit(*circuit.qregs)
                                qfi_circuit.data = circuit.data
                                # Fix ancilla phase
                                sign = np.sign(np.conj(coeff_i)*coeff_j)
                                complex = np.iscomplex(np.conj(coeff_i)*coeff_j)
                                if sign == -1:
                                    if complex:
                                        insert_gate(qfi_circuit, parameterized_gates[params[0]][0], SdgGate(),
                                                    qubits=[ancilla])
                                        # qfi_circuit.sdg(ancilla)
                                    else:
                                        insert_gate(qfi_circuit, parameterized_gates[params[0]][0], ZGate(),
                                                    qubits=[ancilla])
                                        # qfi_circuit.z(ancilla)
                                else:
                                    if complex:
                                        insert_gate(qfi_circuit, parameterized_gates[params[0]][0], SGate(),
                                                    qubits=[ancilla])
                                        # qfi_circuit.s(ancilla)
                                insert_gate(qfi_circuit, parameterized_gates[params[0]][0], XGate(),
                                            qubits=[ancilla])
                                # qfi_circuit.x(ancilla)

                                insert_gate(qfi_circuit, parameterized_gates[params[i]][m], gate_to_insert_i,
                                                                         additional_qubits=additional_qubits)

                                insert_gate(qfi_circuit, gate_to_insert_i, XGate(), qubits=[ancilla], after=True)
                                #
                                # qfi_circuit.x(ancilla)
                                insert_gate(qfi_circuit, parameterized_gates[params[j]][n], gate_to_insert_j,
                                                                         additional_qubits=additional_qubits)

                                # Remove redundant gates
                                qfi_circuit = trim_circuit(qfi_circuit, parameterized_gates[params[i]][m])

                                qfi_circuit.h(ancilla)
                                circuits += [qfi_circuit]
                j += 1
        circuits_phase_fix = []
        for i in range(len(params)):  # loop over parameters
                # create a copy of the original circuit with the same registers
                # circuit = QuantumCircuit(*self._circuit.qregs, qr_ancilla)
                # circuit.data = self._circuit.data
                # insert_gate(circuit, parameterized_gates[params[0]][0], HGate(),
                #                                                qubits=[ancilla])

                # construct the phase fix circuits

                for m, gates_to_insert_i in enumerate(qfi_gates[params[i]]):
                    for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                        qfi_circuit = QuantumCircuit(*circuit.qregs)
                        qfi_circuit.data = circuit.data
                        # insert_gate(qfi_circuit, parameterized_gates[params[0]][0], XGate(), qubits=[ancilla])
                        # Fix ancilla phase
                        insert_gate(qfi_circuit, parameterized_gates[params[i]][m],
                                                                 gate_to_insert_i,
                                                                 additional_qubits=additional_qubits)
                        # insert_gate(qfi_circuit, gate_to_insert_i, XGate(), qubits=[ancilla], after=True)
                        qfi_circuit = trim_circuit(qfi_circuit, parameterized_gates[params[i]][m])

                        # qfi_circuit.h(ancilla)
                        circuits_phase_fix += [qfi_circuit]
        return circuits, circuits_phase_fix

    # @staticmethod
    # def insert_gate(circuit: QuantumCircuit,
    #                 reference_gate: Gate,
    #                 gate_to_insert: Gate,
    #                 qubits: Optional[List[Qubit]] = None,
    #                 additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None) -> bool:
    #     """Insert a gate into the circuit.
    #
    #     Args:
    #         circuit: The circuit onto which the gare is added.
    #         reference_gate: A gate instance before or after which a gate is inserted.
    #         gate_to_insert: The gate to be inserted.
    #         qubits: The qubits on which the gate is inserted. If None, the qubits of the
    #             reference_gate are used.
    #         additional_qubits: If qubits is None and the qubits of the reference_gate are
    #             used, this can be used to specify additional qubits before (first list in
    #             tuple) or after (second list in tuple) the qubits.
    #
    #     Returns:
    #         True, if the insertion has been successful, False otherwise.
    #     """
    #     # TODO add before/after gate again --> @julien sorry
    #     if isinstance(gate_to_insert, IGate):
    #         return True
    #     for i, op in enumerate(circuit.data):
    #         if op[0] == reference_gate:
    #             qubits = qubits or op[1]
    #             if additional_qubits:
    #                 qubits = additional_qubits[0] + qubits + additional_qubits[1]
    #             op_to_insert = (gate_to_insert, qubits, [])
    #             insertion_index = i
    #             circuit.data.insert(insertion_index, op_to_insert)
    #             return True
    #
    #     return False

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

    # @staticmethod
    # def get_coeffs_gates(gate: Gate) -> Tuple[List[complex], List[Gate]]:
    #     """Get the ancilla-controlled gates for the quantum Fisher Information.
    #
    #     Notably, if gate is a two-qubit gate acting on qubits q0, q1 and the returned gate list consists of a gate
    #     tuple (gate0, gate1) then gate0 must act on q0 and gate1 on q1.
    #
    #     Currently, not all parametrized gates are supported.
    #
    #     Args:
    #         gate: The gate for which the derivative is being computed.
    #
    #     Returns:
    #         The coefficients and the gates used for the metric computation for each parameter of the respective gates.
    #
    #     Raises:
    #         TypeError: If the input gate is not a supported parametrized gate.
    #     """
    #
    #     if isinstance(gate, U1Gate):
    #         # theta
    #         return [0.5j, -0.5j], [IGate(), CZGate()]
    #     # TODO Extend to gates with multiple parameters
    #     # if isinstance(gate, U2Gate):
    #     #     # TODO encode multiple parameters
    #     # TODO https://qiskit.org/documentation/_modules/qiskit/aqua/operators/common.html#commutator
    #     #     # theta, phi
    #     #     return [[0.5j], [-0.5j]], [[CZGate], [CZGate]]
    #     # if isinstance(gate, U3Gate):
    #     #     # TODO encode multiple parameters
    #     #     # theta, lambda, phi
    #     #     return [[0.5j], [-0.5j], [+0.5j]], [[CZGate], [CZGate], [CZGate]]
    #     if isinstance(gate, RXGate):
    #         # theta
    #         return [-0.5j], [CXGate()]
    #     if isinstance(gate, RYGate):
    #         # theta
    #         return [-0.5j], [CYGate()]
    #     if isinstance(gate, RZGate):
    #         # theta
    #         # Note that the implemented RZ gate is not an actual RZ gate but [[1, 0], [0, e^i\theta]]
    #         return [-0.5j], [CZGate()]
    #     if isinstance(gate, RXXGate):
    #         # theta
    #         return [-0.5j], [CXGate()] # on both XX - CXX
    #     if isinstance(gate, RYYGate):
    #         # theta
    #         return [-0.5j], [CYGate()] # on both YY - CYY
    #     if isinstance(gate, RZZGate):
    #         # theta
    #         return [-0.5j], [CZGate()] # on both ZZ - CZZ
    #     # TODO wait until this gate is fixed
    #     # if isinstance(gate, RZXGate):
    #     #     # theta
    #     #     return [[-0.5j]], [[(CZXGate]]
    #     if isinstance(gate, CRXGate):
    #         # theta
    #         return [-0.25j, +0.25j], [(IGate(), CXGate()), (CZGate(), CXGate())]
    #     if isinstance(gate, CRYGate):
    #         # theta
    #         return [-0.25j, +0.25j], [(IGate(), CYGate()), (CZGate(), CYGate())]
    #     if isinstance(gate, CRZGate):
    #         # theta
    #         # Note that the implemented RZ gate is not an actual RZ gate but [[1, 0], [0, e^i\theta]]
    #         return [-0.25j, +0.25j], [(IGate(), CZGate()), CZGate()]
    #     if isinstance(gate, CU1Gate):
    #         # theta
    #         return [0.25j, -0.25j, -0.25j, 0.25j], [IGate(), (IGate(), CZGate()), (CZGate(), IGate()),
    #                                                 (CZGate(), CZGate())]
    #
    #     r'''
    #     TODO multi-controlled-$U(\theta)$ for $m$ controlls:
    #     $\frac{1}{2^m}\Bigotimes\limits_i=0^{m-1}(I\-Z)\otimes \frac{\partial U}{\partial\theta} $
    #     # for self.num_ctrl_qubits
    #     '''
    #
    #     raise TypeError('Unrecognized parametrized gate, {}'.format(gate))
    #
    # @staticmethod
    # def trim_circuit(circuit: QuantumCircuit, reference_gate: Gate) -> QuantumCircuit:
    #     """Trim the given quantum circuit before the reference gate.
    #
    #
    #     Args:
    #         circuit: The circuit onto which the gare is added.
    #         reference_gate: A gate instance before or after which a gate is inserted.
    #
    #     Returns:
    #         The trimmed circuit.
    #
    #     Raise:
    #         AquaError: If the reference gate is not part of the given circuit.
    #     """
    #     parameterized_gates = []
    #     for param, elements in circuit._parameter_table.items():
    #         for element in elements:
    #             parameterized_gates.append(element[0])
    #
    #     for i, op in enumerate(circuit.data):
    #         if op[0] == reference_gate:
    #             trimmed_circuit = QuantumCircuit(*circuit.qregs)
    #             trimmed_circuit.data = circuit.data[:i]
    #             return trimmed_circuit
    #
    #     raise AquaError('The reference gate is not in the given quantum circuit.')
