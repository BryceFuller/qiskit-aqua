# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test Quantum Fisher Information """


import unittest
from test.aqua import QiskitAquaTestCase

from qiskit import BasicAer
from qiskit.aqua.operators.gradients.q_fisher_inf import QuantumFisherInf
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.extensions.standard import CRZGate
import numpy as np

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators.gradients.gradients_utils import insert_gate, trim_circuit


# from .gradient import Gradient
class TestQuantumFisherInf(QiskitAquaTestCase):
    """ Test Quantum Fisher Information """
    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 50
        # Set quantum instance to run the quantum generator
        self.qi = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                  seed_simulator=2,
                                  seed_transpiler=2)
        # self.qi_qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'),
        #                                shots=1000,
        #                                seed_simulator=2,
        #                                seed_transpiler=2)
        pass

    def test_qc_trimming(self):
        """Test if quantum circuits are correctly trimmed after a reference gate"""
        p0 = Parameter('p0')
        p1 = Parameter('p1')
        p2 = Parameter('p2')
        p = [p0, p1, p2]
        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.rx(p[0], q[1])
        qc.cry(p[1], q[1], q[0])
        qc.rz(p[2], q[1])
        parameterized_gates = []
        for param, elements in qc._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])
        reference_gate = parameterized_gates[2]
        trimmed_qc = trim_circuit(qc, reference_gate)

        self.assertEqual(trimmed_qc.data, qc.data[:2])

    # def test_qc_inserting(self):
    #     """Test if quantum circuits are correctly trimmed after a reference gate"""
    #     p0 = Parameter('p0')
    #     p1 = Parameter('p1')
    #
    #     q = QuantumRegister(2)
    #     qc = QuantumCircuit(q)
    #     qc.rx(p0, q[1])
    #     qc.cry(p1, q[1], q[0])
    #
    #     parameterized_gates = []
    #     for param, elements in qc._parameter_table.items():
    #         for element in elements:
    #             parameterized_gates.append(element[0])
    #
    #     qr_ancilla = QuantumRegister(1, 'ancilla')
    #     ancilla = qr_ancilla[0]
    #     qc.add_register(qr_ancilla)
    #
    #     additional_qubits = ([ancilla], [])
    #     p_new = Parameter('p')
    #     insert_gate(qc, parameterized_gates[0], CRZGate(p_new), qubits=[q[1]],
    #                                  additional_qubits=additional_qubits)
    #     self.assertTrue(success)

    # def test_construct_circuits(self):
    #     """Test if quantum circuits to be evaluated are constructed"""
    #
    #     p0 = Parameter('p0')
    #     p1 = Parameter('p1')
    #     p2 = Parameter('p2')
    #     p = [p0, p1, p2]
    #     q = QuantumRegister(2)
    #     qc = QuantumCircuit(q)
    #     qc.rx(p[0], q[1])
    #     qc.cry(p[1], q[1], q[0])
    #     qc.rz(p[2], q[1])
    #     parameterized_gates = []
    #     for param, elements in qc._parameter_table.items():
    #         for element in elements:
    #             parameterized_gates.append(element[0])
    #
    #     qfi = QuantumFisherInf(circuit=qc, quantum_instance=self.qi)
    #     qfi_circuits, qfi_phase_fix_circuits = qfi.construct_circuits(parameterized_gates)
    #
    #     self.assertEqual(len(qfi_circuits), 11)
    #     self.assertEqual(len(qfi_phase_fix_circuits), 4)

    def test_qfi(self):
        """Test if the quantum fisher information calculation is correct
        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(p[0])]]"""

        p0 = Parameter('p0')
        p1 = Parameter('p1')
        p = [p0, p1]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(p[0], q[0])
        qc.rx(p[1], q[0])

        parameterized_gates = []
        for param, elements in qc._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        qfi = QuantumFisherInf(circuit=qc, quantum_instance=self.qi)
        values_dict = {p[0]: np.pi / 4, p[1]: 0.1}
        qfi_value=qfi.compute_qfi(p, values_dict)
        correct_qfi = np.allclose(qfi_value, [[1, 0], [0, 0.5]], atol=1e-6)
        values_dict = {p[0]: np.pi, p[1]: 0.1}
        qfi_value = qfi.compute_qfi(p, values_dict)
        correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 0]], atol=1e-6)
        values_dict = {p[0]: np.pi/2, p[1]: 0.1}
        qfi_value = qfi.compute_qfi(p, values_dict)
        correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 1]], atol=1e-6)
        self.assertTrue(correct_qfi)


if __name__ == '__main__':
    unittest.main()
