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

"""The module to compute the state gradient with the linear combination method."""

from typing import Optional, Union, List
from copy import deepcopy
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector
from qiskit.circuit.library import ZGate, SGate, SdgGate, HGate

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import Z
from ..gradient_base import GradientBase


class StateGradientLinComb(GradientBase):
    """Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω using the linear combination method."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None
                ) -> OperatorBase:
        """
        Args:
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp where the ith operator corresponds to the gradient wrt params[i]
        """
        self._params = params
        return self._prepare_operator(operator)

    # TODO remove here. Traverse should happen in gradient
    def _prepare_operator(self, operator):
        if isinstance(operator, ListOp):
            return operator.traverse(self._prepare_operator)
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                return operator.traverse(self._prepare_operator)
        elif isinstance(operator, PrimitiveOp):
            return Z ^ operator
        if isinstance(operator, (QuantumCircuit, CircuitStateFn, CircuitOp)):
            # TODO avoid duplicate transformations
            operator = self._grad_states(operator, self._params)
        return 2 * operator

    def _grad_states(self,
                     op: OperatorBase,
                     target_params: Optional[Union[Parameter, ParameterVector, List]] = None
                     ) -> ListOp:
        """Generate the gradient states.

        Args:
            op: The operator representing the quantum state for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
        """
        # TODO what's with these arguments?
        # state_qc: QuantumCircuit,
        # gates_to_parameters: Dict[Parameter, List[Gate]],
        # grad_coeffs: Dict[Parameter, List[List[complex]]],
        # grad_gates: Dict[Parameter, List[List[Instruction]]])
        # state_qc: The quantum circuit representing the state for which we compute the gradient.
        # gates_to_parameters: The dictionary of parameters and gates with respect to which the
        #   quantum Fisher
        # Information is computed.
        # grad_coeffs: The values needed to compute the gradient for the parameterized gates.
        #      For each parameter, the dict holds a list of all coeffs for all gates which are
        #      parameterized by
        #      the parameter. {param:[[coeffs0],...]}
        # grad_gates: The gates needed to compute the gradient for the parameterized gates.
        #      For each parameter, the dict holds a list of all gates to insert for all gates
        #      which are parameterized by the parameter. {param:[[gates_to_insert0],...]}

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        grad_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        grad_gates = {}
        # Loop throuh the parameters in the circuit
        params = []

        if isinstance(op, (CircuitStateFn, CircuitOp)):
            pass
        elif isinstance(op, (DictStateFn, VectorStateFn)):
            op = DictToCircuitSum().convert(op)  # TODO inplace
        else:
            raise TypeError('Ancilla gradients only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        state_qc = deepcopy(op.primitive)
        for param, elements in state_qc._parameter_table.items():
            # TODO param expressions
            if param not in target_params:
                continue
            if param not in params:
                params.append(param)
                gates_to_parameters[param] = []
            grad_coeffs[param] = []
            grad_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized
                # gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    grad_coeffs[param].append(c_g[0])
                    grad_gates[param].append(c_g[1])

        states = []
        qr_work = QuantumRegister(1, 'work_qubit')
        work_q = qr_work[0]
        additional_qubits = ([work_q], [])
        # create a copy of the original state with an additional work_q register
        # Get the states needed to compute the gradient
        for param in params:  # loop over parameters
            # construct the states
            for m, gates_to_insert_i in enumerate(grad_gates[param]):
                for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                    grad_state = QuantumCircuit(*state_qc.qregs, qr_work)
                    grad_state.compose(state_qc, inplace=True)

                    # apply Hadamard on work_q
                    # TODO can this not just be grad_state.h(qr_work) ?
                    self.insert_gate(
                        grad_state, gates_to_parameters[params[0]][0], HGate(), qubits=[work_q]
                    )

                    # Fix work_q phase
                    coeff_i = grad_coeffs[param][m][k]
                    sign = np.sign(coeff_i)
                    is_complex = np.iscomplex(coeff_i)
                    if sign == -1:
                        if is_complex:
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0],
                                             SdgGate(), qubits=[work_q])
                        else:
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0],
                                             ZGate(), qubits=[work_q])
                    else:
                        if is_complex:
                            self.insert_gate(grad_state, gates_to_parameters[params[0]][0],
                                             SGate(), qubits=[work_q])
                    # Insert controlled, intercepting gate - controlled by |0>
                    self.insert_gate(grad_state, gates_to_parameters[param][m],
                                     gate_to_insert_i,
                                     additional_qubits=additional_qubits)
                    grad_state.h(work_q)
                    if m == 0 and k == 0:
                        state = np.sqrt(np.abs(coeff_i)) * CircuitStateFn(grad_state)
                    else:
                        state += np.sqrt(np.abs(coeff_i)) * CircuitStateFn(grad_state)

            states += [state]
            #  TODO check that all properties of op are carried over but I think so
        return ListOp(states) * op.coeff
