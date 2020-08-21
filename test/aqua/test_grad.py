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


from qiskit.aqua.operators.gradients.gradient.state_gradient_lin_comb import StateGradientLinComb
from qiskit.aqua.operators.gradients.gradient.probability_gradient_lin_comb import ProbabilityGradientLinComb
from qiskit.aqua.operators.gradients.hessian.state_hessian_lin_comb import StateHessianLinComb
from qiskit.aqua.operators.gradients.qfi.qfi import QFI

from qiskit.aqua.operators import X, Z, StateFn, CircuitStateFn

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
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

    def test_state_lin_comb_grad(self):
        """Test the linear combination state gradient
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
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = StateGradientLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                          [-0.5, -1 / np.sqrt(2)]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            correct_grad &= np.allclose(state_grad.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)

        self.assertTrue(correct_grad)

    def test_state_lin_comb_hessian(self):
        """Test the linear combination state Hessian
        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d^2<H>/da^2 = - 0.5 cos(a) + 1 sin(a)sin(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/db^2 = + 1 sin(a)sin(b)
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

        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_hess = StateHessianLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {a: np.pi / 4, b: np.pi / 4},
                       {a: np.pi / 2, b: np.pi / 4}]
        correct_values = [[[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]],
                          [[-0.5 / np.sqrt(2) + 0.5, -1 / 2.], [-0.5, 0.5]],
                          [[1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2)]]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            correct_grad &= np.allclose(state_hess.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)

        self.assertTrue(correct_grad)

    def test_prob_lin_comb_grad(self):
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

        op = CircuitStateFn(primitive=qc, coeff=1.)

        prob_grad = ProbabilityGradientLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: 0}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi}]
        correct_values = [[[0, 0], [1/(2*np.sqrt(2)), - 1/(2*np.sqrt(2))]], [[1/4, - 1/4], [1/4, - 1/4]],
                          [[0, 0], [- 1/2, 1/2]]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            print(prob_grad.assign_parameters(value_dict).eval())
            # print(prob_grad.assign_parameters(value_dict).eval())
            correct_grad &= np.allclose(prob_grad.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)
        self.assertTrue(correct_grad)
    #
    # def test_product_rule(self):
    #     # TODO
    #     return

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

        op = CircuitStateFn(primitive=qc, coeff=1.)
        qfi = QFI().convert(operator=op, params=params)
        values_dict = [{params[0]: np.pi / 4, params[1]: 0.1}, {params[0]: np.pi, params[1]: 0.1},
                       {params[0]: np.pi/2, params[1]: 0.1}]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]],  [[1, 0], [0, 1]]]
        correct_qfi = True
        for i, value_dict in enumerate(values_dict):
            correct_qfi &= np.allclose(qfi.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)
        self.assertTrue(correct_qfi)


if __name__ == '__main__':
    unittest.main()
