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

""" Test Quantum Gradient Framework """


import unittest
from test.aqua import QiskitAquaTestCase

from qiskit import BasicAer

from qiskit.aqua.operators.gradients import Gradient

from qiskit.aqua.operators.gradients.gradient.gradient_lin_comb import GradientLinComb
from qiskit.aqua.operators.gradients.hessian.hessian_lin_comb import HessianLinComb
from qiskit.aqua.operators.gradients.qfi.qfi import QFI

from qiskit.aqua.operators import X, Z, StateFn, CircuitStateFn, ListOp

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterExpression
import numpy as np
from sympy import Symbol, cos

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

    # def test_gradient(self):
    #     """Test the linear combination state gradient
    #     Tr(|psi><psi|Z) = sin(a)sin(b)
    #     Tr(|psi><psi|X) = cos(a)
    #     d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
    #     d<H>/db = - 1 sin(a)cos(b)
    #     """
    #
    #     H = 0.5 * X - 1 * Z
    #     a = Parameter('a')
    #     b = Parameter('b')
    #     params = [a, b]
    #
    #     q = QuantumRegister(1)
    #     qc = QuantumCircuit(q)
    #     qc.h(q)
    #     qc.rz(params[0], q[0])
    #     qc.rx(params[1], q[0])
    #     op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
    #
    #     state_grad = Gradient().convert(operator=op, params=params, method='lin_comb')
    #     values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
    #                    {params[0]: np.pi / 2, params[1]: np.pi / 4}]
    #     correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
    #                       [-0.5, -1 / np.sqrt(2)]]
    #     correct_grad = True
    #     for i, value_dict in enumerate(values_dict):
    #         correct_grad &= np.allclose(state_grad.assign_parameters(
    #             value_dict).eval(), correct_values[i], atol=1e-6)

        # self.assertTrue(correct_grad)

    def test_state_lin_comb_grad(self):
        """Test the linear combination state gradient
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        """
        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = GradientLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                          [-0.5, -1 / np.sqrt(2)]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            correct_grad &= np.allclose(state_grad.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)

        """
           Parameter Expression
           Tr(|psi><psi|Z) = sin(a)sin(a)
           Tr(|psi><psi|X) = cos(a)
           d<H>/da = - 0.5 sin(a) - 2 cos(a)sin(a)
           """
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        # b = Parameter('b')
        params = [a]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(a, q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = GradientLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0},
                       {a: np.pi / 2}]
        correct_values = [[-1.353553], [-0], [-0.5]]

        for i, value_dict in enumerate(values_dict):
            correct_grad &= np.allclose(state_grad.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)

        """
        Parameter Expression
        Tr(|psi><psi|Z) = sin(a)sin(c(a)) = sin(a)sin(cos(a)+1)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(cos(a)+1) + 1 sin^2(a)cos(cos(a)+1)
        """
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        # b = Parameter('b')
        params = [a]
        x = Symbol('x')
        expr = cos(x) + 1
        c = ParameterExpression({a: x}, expr)

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(c, q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = GradientLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0},
                       {a: np.pi / 2}]
        correct_values = [[-1.1220], [-0.9093], [0.0403]]

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

        state_hess = HessianLinComb().convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {a: np.pi / 4, b: np.pi / 4},
                       {a: np.pi / 2, b: np.pi / 4}]
        correct_values = [[[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 0]],
                          [[-0.5 / np.sqrt(2) + 0.5, -1 / 2.], [-0.5, 0.5]],
                          [[1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2)]]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            # TODO coefficient propagation doesn't seem to work
            correct_grad &= np.allclose(state_hess.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)

        self.assertTrue(correct_grad)

    # def test_prob_lin_comb_grad(self):
    #     """Test the ancilla probability gradient
    #     dp0/da = cos(a)sin(b) / 2
    #     dp1/da = - cos(a)sin(b) / 2
    #     dp0/db = sin(a)cos(b) / 2
    #     dp1/db = - sin(a)cos(b) / 2
    #     """
    #
    #     a = Parameter('a')
    #     b = Parameter('b')
    #     params = [a, b]
    #
    #     q = QuantumRegister(1)
    #     qc = QuantumCircuit(q)
    #     qc.h(q)
    #     qc.rz(params[0], q[0])
    #     qc.rx(params[1], q[0])
    #
    #     op = CircuitStateFn(primitive=qc, coeff=1.)
    #
    #     prob_grad = GradientLinComb().convert(operator=op, params=params)
    #     values_dict = [{a: np.pi / 4, b: 0}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
    #                    {params[0]: np.pi / 2, params[1]: np.pi}]
    #     correct_values = [[[0, 0], [1/(2*np.sqrt(2)), - 1/(2*np.sqrt(2))]], [[1/4, - 1/4], [1/4, - 1/4]],
    #                       [[0, 0], [- 1/2, 1/2]]]
    #     correct_grad = True
    #     for i, value_dict in enumerate(values_dict):
    #         print(prob_grad.assign_parameters(value_dict).eval())
    #         # print(prob_grad.assign_parameters(value_dict).eval())
    #         correct_grad &= np.allclose(prob_grad.assign_parameters(
    #             value_dict).eval(), correct_values[i], atol=1e-6)
    #     self.assertTrue(correct_grad)
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

    def test_jax_chain_rule(self):
        """Test that the chain rule functionality using Jax"""
        def combo_fn(x):
            return x[0]**2 + cos(x[1])
        H = ListOp([X, Z], combo_fn=combo_fn)
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        """
        d<H>/d<X> = 2<X>
        d<H>/d<X> = - sin(<Z>)
        <Z> = Tr(|psi><psi|Z) = sin(a)sin(b)
        <X> = Tr(|psi><psi|X) = cos(a)
        d<H>/da = d<H>/d<X> d<X>/da + d<H>/d<Z> d<Z>/da = - 2 cos(a)sin(a) + sin(sin(a)sin(b)) * cos(a)sin(b)
        d<H>/db = d<H>/d<X> d<X>/db + d<H>/d<Z> d<Z>/db = sin(sin(a)sin(b)) * sin(a)cos(b)
        """

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient().convert(operator=op, params=params, method='lin_comb')
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-1., 0.], [-0.76029, 0.2397], [0., 0.45936]]
        correct_grad = True
        for i, value_dict in enumerate(values_dict):
            correct_grad &= np.allclose(state_grad.assign_parameters(
                value_dict).eval(), correct_values[i], atol=1e-6)
        self.assertTrue(correct_grad)


if __name__ == '__main__':
    unittest.main()
