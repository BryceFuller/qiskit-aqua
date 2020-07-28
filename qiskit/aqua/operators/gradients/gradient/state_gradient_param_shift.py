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

""" StateGradient Class """

from typing import Optional, Callable, Union, List, Dict
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Gate

from qiskit.aqua.operators import OperatorBase, ListOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.gradients import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.circuit.library.standard_gates import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, CZGate,\
    U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, XGate, \
    SdgGate, SGate, ZGate

logger = logging.getLogger(__name__)


class StateGradientParamShift(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for ω in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None,
                analytic: bool = True) -> OperatorBase:
        r"""
        Args
            state_operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            observable_operator: The measurement operator we are taking the gradient of: O(θ)
            params: The parameters we are taking the gradient wrt: ω
            analytic: If true compute an analytic gradient, else compute a finite difference approximations
        Returns
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        # TODO add finite difference

        # TODO Look through state and decompose gates which cannot be evaluated with the parameter shift rule
        # This seems like it could be it's own converter??
        # decomposed_state = self.decompose_to_two_unique_eigenval(state_operator, params)
        # TODO Note to above --> all are pi/2
        # parameter_shift will return a ListOp of the same size as params
        # one SummedOp per parameter.
        return self.parameter_shift(operator, params)

