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

""" StateHessian Class """

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

from qiskit.circuit.library.standard_gates import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, \
    CZGate, \
    U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, XGate, \
    SdgGate, SGate, ZGate

logger = logging.getLogger(__name__)


class StateHessianAncilla(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for ω in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            state_operator:The operator corresponding to our quantum state we are taking the hessian of: |ψ(ω)〉
            observable_operator: The measurement operator we are taking the hessian of: O(θ)
            params: The parameters we are taking the hessian wrt: ω
        Returns
            ListOp where the ith operator corresponds to the hessian wrt params[i]
        """

        # TODO split operator in state and observable

        grad_op = StateFn(2 * observable_operator ^ Z).adjoint()  # Prefactor needed for correction
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        grad_gates = {}
        # Loop throuh the parameters in the circuit
        params = []
        if isinstance(state_operator, CircuitStateFn):
            state_qc = state_operator.primitive
        elif isinstance(state_operator, DictStateFn) or isinstance(state_operator, VectorStateFn):
            state_qc = DictToCircuitSum.convert(state_operator).primitive
        else:
            raise TypeError('Ancilla hessians only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        for param, elements in state_qc._parameter_table.items():
            params.append(param)
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_hessian_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    grad_coeffs[param].append(c_g[0])
                    grad_gates[param].append(c_g[1])

        states = self._ancilla_grad_states(gates_to_parameters, grad_coeffs, grad_gates)

        return grad_op @ states


def _ancilla_hessian_states(self, parameterized_gates: Dict[Parameter, List[Gate]],
                       grad_coeffs: Dict[Parameter, List[List[complex]]],
                       grad_gates: Dict[Parameter, List[List[Instruction]]]) -> \
        List[QuantumCircuit]:
    """Generate the hessian states.

    Args:
        parameterized_gates: The dictionary of parameters and gates with respect to which the quantum Fisher
        Information is computed.
        grad_coeffs: The values needed to compute the hessian for the parameterized gates.
                For each parameter, the dict holds a list of all coeffs for all gates which are parameterized by
                the parameter. {param:[[coeffs0],...]}
        grad_gates: The gates needed to compute the hessian for the parameterized gates.
                For each parameter, the dict holds a list of all gates to insert for all gates which are
                parameterized by the parameter. {param:[[gates_to_insert0],...]}

    Returns:
        List of quantum circuits which are needed to compute the hessian.
        If a parameter appears multiple times, one circuit is created per parameterized gates to be able to compute
        the product rule.

    Raises:
        AquaError: If one of the circuits could not be constructed.
    """

    states = []
    qr_ancilla = QuantumRegister(1, 'ancilla')
    ancilla = qr_ancilla[0]
    additional_qubits = ([ancilla], [])
    # create a copy of the original state with an additional ancilla register
    state = QuantumCircuit(*self._state.qregs, qr_ancilla)
    state.data = self._state.data
    params = list(parameterized_gates.keys())
    # apply Hadamard on ancilla
    insert_gate(state, parameterized_gates[params[0]][0], HGate(),
                qubits=[ancilla])
    # Get the states needed to compute the hessian
    for i in range(len(params)):  # loop over parameters
        # construct the states
        for m, gates_to_insert_i in enumerate(grad_gates[params[i]]):
            for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                grad_state = QuantumCircuit(*state.qregs)
                grad_state.data = state.data

                # Fix ancilla phase
                coeff_i = grad_coeffs[params[i]][m][k]
                sign = np.sign(coeff_i)
                complex = np.iscomplex(coeff_i)
                if sign == -1:
                    if complex:
                        insert_gate(grad_state, parameterized_gates[params[0]][0], SdgGate(),
                                    qubits=[ancilla])
                    else:
                        insert_gate(grad_state, parameterized_gates[params[0]][0], ZGate(),
                                    qubits=[ancilla])
                else:
                    if complex:
                        insert_gate(grad_state, parameterized_gates[params[0]][0], SGate(),
                                    qubits=[ancilla])
                # Insert controlled, intercepting gate - controlled by |0>
                insert_gate(grad_state, parameterized_gates[params[i]][m],
                            gate_to_insert_i,
                            additional_qubits=additional_qubits)
                grad_state.h(ancilla)
                states += [grad_state]
    return states
