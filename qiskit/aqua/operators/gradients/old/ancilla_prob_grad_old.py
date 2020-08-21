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

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile

from qiskit.circuit import Parameter, Gate, Qubit, Instruction

from qiskit.aqua.operators import PauliOp, Z, I, CircuitSampler, StateFn, OperatorBase
from qiskit.quantum_info import partial_trace
from qiskit.aqua.utils.run_circuits import find_regs_by_name

from qiskit.circuit.library.standard_gates import HGate, SdgGate, SGate, ZGate

from qiskit.aqua import QuantumInstance

from qiskit.aqua.operators.gradients.gradient.probability_gradient import Gradient

from .grad_utils import gate_gradient_dict, insert_gate


class AncillaProbGradient(Gradient):
    """Compute the gradient of a pure, parametrized quantum state using an ancilla and intercepting,
    controlled gates."""

    def __init__(self, circuit: Optional[QuantumCircuit] = None,
                 quantum_instance: Optional[QuantumInstance] = None) -> None:
        """
        Args:
            circuit: The circuit for which the quantum Fisher information is computed.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        super().__init__(circuit, quantum_instance=quantum_instance)

        warnings.simplefilter("once")

    def compute_grad(self, parameters: Parameter, parameter_values: Dict,
                     grad_circuits: Optional[Tuple[List[QuantumCircuit], List[QuantumCircuit]]] = None) -> np.ndarray:
        """Compute the entry of gradient with respect to the provided parameters.

        Args:
            parameters: The parameters with respect to which the gradient is computed.
            parameter_values: The values of the parameters with respect to which the gradient
            is computed.
            grad_circuits: Tuple of lists of quantum circuits needed to compute the gradient.

        Returns: probability gradient array of shape (num params, num qubits)
        """

        def get_prob_value(circuit: List[QuantumCircuit], operator: OperatorBase = Z) -> List[List[float]]:
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

                return qubit_op, index_evaluation_qubit

            if not isinstance(circuit, list):
                circuit = [circuit]
            results = []
            for k, circuit_item in enumerate(circuit):
                # Transpile & assign parameter values
                circuit_item = transpile(circuit_item, backend=self._quantum_instance.backend)
                new_dict = {param: value for param, value in master_dict.items() if
                            param in circuit_item.parameters}
                circuit_item = circuit_item.assign_parameters(new_dict)
                # Get operator for partial expectation value
                op, index_evaluation_qubit = prepare(circuit_item)
                op = op.to_matrix()
                # Construct circuits to evaluate the expectation values
                sampler = CircuitSampler(self._quantum_instance).convert(StateFn(circuit_item))
                result = sampler.to_density_matrix()
                prob_grad = partial_trace(op.dot(result), [index_evaluation_qubit])
                results.append(list(np.diag(prob_grad.data)))
            return results

        master_dict = parameter_values

        grad = np.zeros((len(parameters), 2**self.circuit.num_qubits), dtype=complex)
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        grad_gates = {}
        # Loop throuh the parameters in the circuit
        params = []
        for param, elements in self._circuit._parameter_table.items():
            params.append(param)
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    grad_coeffs[param].append(c_g[0])
                    grad_gates[param].append(c_g[1])
        if grad_circuits is None:
            grad_circuits = self.construct_circuits(gates_to_parameters, grad_coeffs, grad_gates)
        else:
            grad_circuits = grad_circuits
        if len(grad_circuits) > 0:
            grad_prob_values = np.array(get_prob_value(grad_circuits))

        counter = 0
        # weighted sum of the circuit expectation values w.r.t. the coefficients
        for i in range(len(params)):
            for coeffs_i in grad_coeffs[params[i]]:
                for coeff_i in coeffs_i:
                    # Quantum circuit already considers sign and if complex.
                    grad[i] += np.abs(coeff_i) * grad_prob_values[counter]
                    counter += 1
        # Add correct pre-factor and return
        return 2*grad

    def construct_circuits(self, parameterized_gates: Dict[Parameter, List[Gate]],
                           grad_coeffs: Dict[Parameter, List[List[complex]]],
                           grad_gates: Dict[Parameter, List[List[Instruction]]]) -> \
            List[QuantumCircuit]:
        """Generate the gradient circuits.

        Args:
            parameterized_gates: The dictionary of parameters and gates with respect to which the quantum Fisher
            Information is computed.
            grad_coeffs: The values needed to compute the gradient for the parameterized gates.
                    For each parameter, the dict holds a list of all coeffs for all gates which are parameterized by
                    the parameter. {param:[[coeffs0],...]}
            grad_gates: The gates needed to compute the gradient for the parameterized gates.
                    For each parameter, the dict holds a list of all gates to insert for all gates which are
                    parameterized by the parameter. {param:[[gates_to_insert0],...]}

        Returns:
            List of quantum circuits which are needed to compute the gradient.
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
        for i in range(len(params)):  # loop over parameters
            # construct the circuits
            for m, gates_to_insert_i in enumerate(grad_gates[params[i]]):
                for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                    grad_circuit = QuantumCircuit(*circuit.qregs)
                    grad_circuit.data = circuit.data

                    # Fix ancilla phase
                    coeff_i = grad_coeffs[params[i]][m][k]
                    sign = np.sign(coeff_i)
                    complex = np.iscomplex(coeff_i)
                    if sign == -1:
                        if complex:
                            insert_gate(grad_circuit, parameterized_gates[params[0]][0], SdgGate(),
                                        qubits=[ancilla])
                        else:
                            insert_gate(grad_circuit, parameterized_gates[params[0]][0], ZGate(),
                                        qubits=[ancilla])
                    else:
                        if complex:
                            insert_gate(grad_circuit, parameterized_gates[params[0]][0], SGate(),
                                        qubits=[ancilla])
                    # if complex:
                    #     insert_gate(grad_circuit, parameterized_gates[params[0]][0], SGate(),
                    #                         qubits=[ancilla])
                    # Insert controlled, intercepting gate - controlled by |0>
                    insert_gate(grad_circuit, parameterized_gates[params[i]][m],
                                gate_to_insert_i,
                                additional_qubits=additional_qubits)
                    grad_circuit.h(ancilla)
                    circuits += [grad_circuit]
        return circuits
