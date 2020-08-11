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
from copy import deepcopy

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Gate

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, PauliOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z
from qiskit.aqua.operators.expectations import PauliExpectation
from ..gradient_base import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.circuit.library.standard_gates import RXGate, CRXGate, RYGate, CRYGate, RZGate, CRZGate, CXGate, CYGate, \
    CZGate,U1Gate, U2Gate, U3Gate, RXXGate, RYYGate, RZZGate, RZXGate, CU1Gate, MCU1Gate, CU3Gate, IGate, HGate, \
    XGate, SdgGate, SGate, ZGate

logger = logging.getLogger(__name__)


class StateGradientAncilla(GradientBase):
    r"""
    We are interested in computing:
    d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω  for ω in params
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
        Returns
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """

        return self._prepare_operator(operator, params)

    def _prepare_operator(self, operator, params):
        if isinstance(operator, ListOp):
            return operator.traverse(self.prepare_operator)
        elif isinstance(operator, StateFn):
            if operator.is_measurement == True:
                return operator.traverse(self.prepare_operator)
        elif isinstance(operator, PrimitiveOp):
            return 2 * (operator ^ Z)  # Z needs to be at the end
        if isinstance(operator, (QuantumCircuit, CircuitStateFn, CircuitOp)):
            # TODO avoid duplicate transformations
            operator = self._grad_states(operator, params)
        return operator

    def _grad_states(self,
                     op: OperatorBase,
                     target_params: Union[Parameter, ParameterVector, List] = None) -> ListOp[CircuitStateFn]:
                             # state_qc: QuantumCircuit,
                             # gates_to_parameters: Dict[Parameter, List[Gate]],
                             # grad_coeffs: Dict[Parameter, List[List[complex]]],
                             # grad_gates: Dict[Parameter, List[List[Instruction]]])
        """Generate the gradient states.

        Args:
            op: The operator representing the quantum state for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω
            state_qc: The quantum circuit representing the state for which we compute the grad.
            gates_to_parameters: The dictionary of parameters and gates with respect to which the quantum Fisher
            Information is computed.
            grad_coeffs: The values needed to compute the gradient for the parameterized gates.
                    For each parameter, the dict holds a list of all coeffs for all gates which are parameterized by
                    the parameter. {param:[[coeffs0],...]}
            grad_gates: The gates needed to compute the gradient for the parameterized gates.
                    For each parameter, the dict holds a list of all gates to insert for all gates which are
                    parameterized by the parameter. {param:[[gates_to_insert0],...]}

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the gradient.
            If a parameter appears multiple times, one circuit is created per parameterized gates to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        grad_gates = {}
        # Loop throuh the parameters in the circuit
        params = []

        if isinstance(op, CircuitStateFn) or isinstance(op, CircuitOp):
            pass
        elif isinstance(op, DictStateFn) or isinstance(op, VectorStateFn):
            op = DictToCircuitSum.convert(op) #Todo inplace
        else:
            raise TypeError('Ancilla gradients only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        state_qc = deepcopy(op.primitive)
        for param, elements in state_qc._parameter_table.items():
            if param not in target_params:
                continue
            params.append(param)
            gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    grad_coeffs[param].append(c_g[0])
                    grad_gates[param].append(c_g[1])

        states = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        additional_qubits = ([ancilla], [])
        # create a copy of the original state with an additional ancilla register
        state = QuantumCircuit(*state_qc.qregs, qr_ancilla)
        state.data = state_qc.data
        params = list(gates_to_parameters.keys())
        # apply Hadamard on ancilla
        self.insert_gate(state, gates_to_parameters[params[0]][0], HGate(),
                    qubits=[ancilla])
        # Get the states needed to compute the gradient
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
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0], SdgGate(),
                                        qubits=[ancilla])
                        else:
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0], ZGate(),
                                        qubits=[ancilla])
                    else:
                        if complex:
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0], SGate(),
                                        qubits=[ancilla])
                    # Insert controlled, intercepting gate - controlled by |0>
                    self.insert_gate(grad_state, gates_to_parameters[params[i]][m],
                                gate_to_insert_i,
                                additional_qubits=additional_qubits)
                    grad_state.h(ancilla)
                    if m == 0 and k == 0:
                        state = np.abs(coeff_i) * CircuitStateFn(grad_state)
                    else:
                        state += np.abs(coeff_i) * CircuitStateFn(grad_state)
            states += [state]
            #  TODO check that all properties of op are carried over but I think so
        return ListOp(states) * op.coeff

