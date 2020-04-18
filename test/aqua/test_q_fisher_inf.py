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
from qiskit.aqua.algorithms import QuantumFisherInf
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate, Qubit
from qiskit.extensions.standard import RXGate, RYGate, RZGate, CRZGate
import numpy as np

from qiskit.aqua import QuantumInstance, aqua_globals


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
        trimmed_qc = QuantumFisherInf.trim_circuit(qc, reference_gate)

        self.assertEqual(trimmed_qc.data, qc.data[:2])

    def test_qc_inserting(self):
        """Test if quantum circuits are correctly trimmed after a reference gate"""
        p0 = Parameter('p0')
        p1 = Parameter('p1')

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.rx(p0, q[1])
        qc.cry(p1, q[1], q[0])

        parameterized_gates = []
        for param, elements in qc._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        qc.add_register(qr_ancilla)

        additional_qubits = ([ancilla], [])
        p_new = Parameter('p')
        success = QuantumFisherInf.insert_gate(qc, parameterized_gates[0],
                                     CRZGate(p_new), qubits=[q[1]],
                                     additional_qubits=additional_qubits)
        #
        # q_test = QuantumRegister(3)
        # qc_test = QuantumCircuit(q_test)
        # qc_test.rx(p0, q_test[1])
        # qc_test.crz(p_new, q_test[2], q_test[1])
        # qc_test.cry(p1, q_test[1], q_test[0])
        #
        # print(qc)
        # print(qc_test)

        self.assertTrue(success)
        # self.assertEqual(qc.data, qc_test.data)

    def test_construct_circuits(self):
        """Test if quantum circuits to be evaluated are constructed"""
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

        qfi = QuantumFisherInf(circuit=qc, quantum_instance=self.qi)
        qfi_circuits = qfi.construct_circuits(parameterized_gates)

        self.assertEqual(len(qfi_circuits), 5)

    def test_qfi(self):
        """Test if the quantum fisher information calculation is correct"""
        # TODO insert correct values and gates for the computed examples.
        """Test if quantum circuits to be evaluated are constructed"""
        # p0 = Parameter('p0')
        # p1 = Parameter('p1')
        # p2 = Parameter('p2')
        # p = [p0, p1, p2]
        # q = QuantumRegister(2)
        # qc = QuantumCircuit(q)
        # qc.rx(p[0], q[1])
        # qc.cry(p[1], q[1], q[0])
        # qc.rz(p[2], q[1])
        # p0 = Parameter('p0')
        # p = [p0]
        # q = QuantumRegister(1)
        # qc = QuantumCircuit(q)
        # qc.ry(p[0], q[0])


        p0 = Parameter('p0')
        p1 = Parameter('p1')
        p = [p0, p1]
        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.rx(p[0], q[1])
        qc.cry(p[1], q[0], q[1])
        values = [0.1, np.pi]

        parameterized_gates = []
        for param, elements in qc._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        qfi = QuantumFisherInf(circuit=qc, quantum_instance=self.qi)
        almost_equal = np.allclose(qfi.compute_qfi(p, values), [[1, 0], [0, 0.5]], atol=1e-10)
        self.assertTrue(almost_equal)

if __name__ == '__main__':
    unittest.main()
