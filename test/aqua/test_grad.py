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
from qiskit.aqua.components.gradients import QFI, AncillaProbGradient, AncillaStateGradient
from qiskit.aqua.operators import X, Z
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import numpy as np

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.gradients.grad_utils import trim_circuit


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

    def test_state_prob_grad(self):
        """Test the ancilla state gradient
        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        state_grad = AncillaStateGradient(circuit=qc, observable=H, quantum_instance=self.qi)
        values_dict = {params[0]: np.pi / 4, params[1]: np.pi}
        grad_value = state_grad.compute_grad(params, values_dict)
        correct_grad = np.allclose(grad_value, [-0.5/np.sqrt(2), 1/np.sqrt(2)], atol=1e-6)
        values_dict = {params[0]: np.pi / 4, params[1]: np.pi / 4}
        grad_value = state_grad.compute_grad(params, values_dict)
        correct_grad &= np.allclose(grad_value, [-1/2 * (1/np.sqrt(2) + 1.), -1/2.], atol=1e-6)
        values_dict = {params[0]: np.pi / 2, params[1]: np.pi / 4}
        grad_value = state_grad.compute_grad(params, values_dict)
        correct_grad &= np.allclose(grad_value, [-0.5, -1/np.sqrt(2)], atol=1e-6)
        self.assertTrue(correct_grad)

    def test_ancilla_prob_grad(self):
        """Test the ancilla probability gradient
        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        prob_grad = AncillaProbGradient(circuit=qc, quantum_instance=self.qi)
        values_dict = {params[0]: np.pi / 4, params[1]: 0}
        grad_value = prob_grad.compute_grad(params, values_dict)
        correct_grad = np.allclose(grad_value, [[0, 0], [1/(2*np.sqrt(2)), - 1/(2*np.sqrt(2))]], atol=1e-6)
        values_dict = {params[0]: np.pi/4, params[1]: np.pi/4}
        grad_value = prob_grad.compute_grad(params, values_dict)
        correct_grad &= np.allclose(grad_value, [[1/4, - 1/4], [1/4, - 1/4]], atol=1e-6)
        values_dict = {params[0]: np.pi/2, params[1]: np.pi}
        grad_value = prob_grad.compute_grad(params, values_dict)
        correct_grad &= np.allclose(grad_value, [[0, 0], [- 1/2, 1/2]], atol=1e-6)
        self.assertTrue(correct_grad)

    def test_product_rule(self):
        # TODO
        return

    def test_qfi(self):
        """Test if the quantum fisher information calculation is correct
        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]"""

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        # parameterized_gates = []
        # for param, elements in qc._parameter_table.items():
        #     for element in elements:
        #         parameterized_gates.append(element[0])

        qfi = QFI(circuit=qc, quantum_instance=self.qi)
        values_dict = {params[0]: np.pi / 4, params[1]: 0.1}
        qfi_value=qfi.compute_qfi(params, values_dict)
        correct_qfi = np.allclose(qfi_value, [[1, 0], [0, 0.5]], atol=1e-6)
        values_dict = {params[0]: np.pi, params[1]: 0.1}
        qfi_value = qfi.compute_qfi(params, values_dict)
        correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 0]], atol=1e-6)
        values_dict = {params[0]: np.pi/2, params[1]: 0.1}
        qfi_value = qfi.compute_qfi(params, values_dict)
        correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 1]], atol=1e-6)
        self.assertTrue(correct_qfi)


if __name__ == '__main__':
    unittest.main()
