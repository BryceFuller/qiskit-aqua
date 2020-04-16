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

from typing import Optional, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Gate, ControlledGate, Qubit
from qiskit.extensions.standard import RXGate, RYGate, RZGate
from copy import deepcopy

from qiskit.aqua import QuantumInstance, AquaError


# from .gradient import Gradient

# p0 = Parameter('p0')
# p1 = Parameter('p1')
# p2 = Parameter('p2')
# p = [p0, p1]
# q = QuantumRegister(2)
# qc = QuantumCircuit(q)
# qc.rx(p[0], q[1])
# qc.cry(p2, q[1], q[0])
# qc.rz(p[1], q[1])
# parameterized_gates = []
# for param, elements in qc._parameter_table.items():
#     for element in elements:
#         parameterized_gates.append(element[0])
# print(parameterized_gates)
# reference_gate = parameterized_gates[1]
# print(reference_gate)
# for i, op in enumerate(qc.data):
#     if op[0] == reference_gate:
#         print(op[1])
#         circuit = QuantumCircuit(*qc.qregs)
#         circuit.data = qc.data[:i+1]
#         print(circuit)

"""
Test trimming
Construct Quantum Circuit with 3 gates
Cut after the second gate
Validate that the first 2 gates are the only ones in qc.data
"""

"""
Test inserting
Construct Quantum Circuit with 2 gates
Insert another gate after the first gate
Validate that the all 3 gates are in qc.data
"""


"""
Test construct circuits
Ensure that the method returns a list of circuits and that the length of the list is correct.
Something else?
"""



"""
Test qfi
Use two test cases. Computed by Amira to check if the results are correct
"""