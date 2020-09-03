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

"""The module for Quantum the Fisher Information."""
from collections.abc import Iterable
from typing import List, Union, Optional
import copy
from copy import deepcopy
from functools import cmp_to_key

import numpy as np
from scipy.linalg import block_diag

from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import RYGate, RZGate, RXGate, HGate, XGate, SdgGate, SGate, ZGate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, CircuitSampler
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.aqua.operators.operator_globals import I, Z, Y, X, Zero

from ..gradient_base import GradientBase
from ...expectations import PauliExpectation


class QFI(GradientBase):
    r"""Compute the Quantum Fisher Information (QFI) given a pure, parametrized quantum state.

    The QFI is:

        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 4.
    """

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[Parameter, ParameterVector, List[Parameter]]] = None,
                approx: Optional[str] = None) -> OperatorBase:
        r"""
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute the QFI
            params: The parameters we are computing the QFI wrt: ω
            approx: Which approximation of the QFI to use: [None, 'diagonal', 'block_diagonal']

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl

        Raises:
            ValueError: If the value for ``approx`` is not supported.
        """

        #TODO: if a ComposedOp is given, try to extract the CircuitStateFn from it and discard
        # the measurement. Throw a warning though. 


        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        if approx is None:
            return self._full_qfi(operator, params)
        elif approx == 'diagonal':
            return self.diagonal_qfi(operator, params)
        elif approx == 'block_diagonal':
            return self.block_diagonal_qfi(operator, params)
        else:
            raise ValueError("Unrecognized input provided for approx. Valid inputs are "
                             "[None, 'diagonal', 'block_diagonal'].")

    def gradient_wrapper(self,
                         operator: OperatorBase,
                         params: Union[Parameter, ParameterVector, List[Parameter]],
                         approx: Optional[str] = None,
                         backend: Optional[Union[BaseBackend, QuantumInstance]] = None) -> callable(Iterable):
        """
        Get a callable function which provides the respective gradient, Hessian or QFI for given parameter values.
        This callable can be used as gradient function for optimizers.
        Args:
            operator: The operator for which we want to get the gradient, Hessian or QFI.
            params: The parameters with respect to which we are taking the gradient, Hessian or QFI.
            approx: Which approximation of the QFI to use: [None, 'diagonal', 'block_diagonal']
            backend: The quantum backend or QuantumInstance to use to evaluate the gradient, Hessian or QFI.
        Returns:
            callable(param_values): Function to compute a gradient, Hessian or QFI. The function takes an Iterable
            as argument which holds the parameter values.

        """

        if not backend:
            converter = self.convert(operator, params, approx)
        else:
            if isinstance(backend, QuantumInstance):
                if backend.is_statevector:
                    converter = self.convert(operator, params, approx)
                else:
                    converter = CircuitSampler(backend=backend).convert(self.convert(operator, params, approx))
            else:
                if backend.name().startswith('statevector'):
                    converter = self.convert(operator, params, approx)
                else:
                    converter = CircuitSampler(backend=backend).convert(self.convert(operator, params, approx))
        return lambda p_values: converter.bind_params(dict(zip(params, p_values))).eval()

    def _full_qfi(self, op: OperatorBase,
                  target_params: Union[Parameter, ParameterVector, List] = None) -> ListOp:
        """Generate the operators whose evaluation leads to the full QFI.

        Args:
            op: The operator representing the quantum state for which we compute the QFI.
            target_params: The parameters we are computing the QFI wrt: ω

        Returns:
            Operators which give the QFI. If a parameter appears multiple times, one circuit is
            created per parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If ``operator`` is an unsupported type.
        """
        # QFI & phase fix observable
        qfi_observable = ~StateFn(4 * Z ^ (I ^ op.num_qubits))
        phase_fix_observable = ~StateFn((X + 1j * Y) ^ (I ^ op.num_qubits))
        # see https://arxiv.org/pdf/quant-ph/0108146.pdf
        # Alternatively, define one operator which computes the QFI with phase fix directly
        # qfi_observable = ~StateFn(Z ^ (I ^ op.num_qubits) - op)

        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        qfi_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        qfi_gates = {}

        if isinstance(op, CircuitStateFn):
            pass
        else:
            raise TypeError('The gradient framework is compatible with states that are given as CircuitStateFn')

        if not isinstance(target_params, Iterable):
            target_params = [target_params]
        state_qc = copy.deepcopy(op.primitive)
        for param in target_params:
            elements = state_qc._parameter_table[param]
            gates_to_parameters[param] = []
            qfi_coeffs[param] = []
            qfi_gates[param] = []
            for element in elements:
                # Get the coefficients and controlled gates (raises an error if the parameterized
                # gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                # Get the gates which are parameterized by param
                gates_to_parameters[param].append(element[0])
                c = []
                g = []
                for j, gate_param in enumerate(element[0].params):
                    c.extend(coeffs_gates[j][0])
                    g.extend(coeffs_gates[j][1])
                qfi_coeffs[param].append(c)
                qfi_gates[param].append(g)

        # First, the operators are computed which can compensate for a potential phase-mismatch
        # between target and trained state, i.e.〈ψ|∂lψ〉
        phase_fix_states = []
        qr_work = QuantumRegister(1, 'work_qubit')
        work_q = qr_work[0]
        additional_qubits = ([work_q], [])
        # create a copy of the original state with an additional work_q register
        for param in target_params:
            for m, gates_to_insert_i in enumerate(qfi_gates[param]):
                for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                    grad_state = QuantumCircuit(*state_qc.qregs, qr_work)
                    grad_state.data = state_qc.data
                    # apply Hadamard on work_q  # TODO can this not just be grad_state.h(work_q)?
                    self.insert_gate(grad_state, gates_to_parameters[target_params[0]][0], HGate(),
                                     qubits=[work_q])
                    # Fix work_q phase
                    coeff_i = qfi_coeffs[param][m][k]
                    sign = np.sign(coeff_i)
                    is_complex = np.iscomplex(coeff_i)
                    if sign == -1:
                        if is_complex:
                            self.insert_gate(grad_state,
                                             gates_to_parameters[target_params[0]][0],
                                             SdgGate(),
                                             qubits=[work_q])
                        else:
                            self.insert_gate(grad_state,
                                             gates_to_parameters[target_params[0]][0],
                                             ZGate(),
                                             qubits=[work_q])
                    else:
                        if is_complex:
                            self.insert_gate(grad_state,
                                             gates_to_parameters[target_params[0]][0],
                                             SGate(),
                                             qubits=[work_q])
                    # Insert controlled, intercepting gate - controlled by |0>
                    self.insert_gate(grad_state, gates_to_parameters[param][m],
                                     gate_to_insert_i,
                                     additional_qubits=additional_qubits)

                    grad_state = self.trim_circuit(grad_state, gates_to_parameters[param][m])

                    grad_state.h(work_q)

                    state = np.sqrt(np.abs(coeff_i)) * \
                                                  op.coeff * CircuitStateFn(grad_state)

                    # Chain Rule parameter expressions
                    gate_param = gates_to_parameters[param][m].params[k]
                    if gate_param == param:
                        state = phase_fix_observable @ state
                    else:
                        if isinstance(gate_param, ParameterExpression):
                            import sympy as sy
                            expr_grad = self.parameter_expression_grad(gate_param, param)
                            state = (expr_grad * phase_fix_observable) @ state
                        else:
                            state *= 0

                    if m == 0 and k == 0:
                        phase_fix_state = state
                    else:
                        phase_fix_state +=  state
            phase_fix_states += [phase_fix_state]

        # Get  4 * Re[〈∂kψ|∂lψ]
        qfi_operators = []
        qr_work_qubit = QuantumRegister(1, 'work_qubit')
        work_qubit = qr_work_qubit[0]
        additional_qubits = ([work_qubit], [])
        # create a copy of the original circuit with an additional work_qubit register
        circuit = QuantumCircuit(*state_qc.qregs, qr_work_qubit)
        circuit.data = state_qc.data
        # apply Hadamard on work_qubit
        self.insert_gate(circuit, gates_to_parameters[target_params[0]][0], HGate(), qubits=[work_qubit])
        # Get the circuits needed to compute A_ij
        for i, param_i in enumerate(target_params):  # loop over parameters
            qfi_ops = []
            for j, param_j in enumerate(target_params):

                # construct the circuits
                for m_i, gates_to_insert_i in enumerate(qfi_gates[param_i]):
                    for k_i, gate_to_insert_i in enumerate(gates_to_insert_i):
                        coeff_i = qfi_coeffs[param_i][m_i][k_i]
                        for m_j, gates_to_insert_j in enumerate(qfi_gates[param_j]):
                            for k_j, gate_to_insert_j in enumerate(gates_to_insert_j):
                                coeff_j = qfi_coeffs[param_j][m_j][k_j]
                                # create a copy of the original circuit with the same registers
                                qfi_circuit = QuantumCircuit(*circuit.qregs)
                                qfi_circuit.data = circuit.data

                                # Fix work_qubit phase
                                sign = np.sign(np.conj(coeff_i) * coeff_j)
                                is_complex = np.iscomplex(np.conj(coeff_i) * coeff_j)
                                if sign == -1:
                                    if is_complex:
                                        self.insert_gate(qfi_circuit,
                                                         gates_to_parameters[target_params[0]][0],
                                                         SdgGate(),
                                                         qubits=[work_qubit])
                                    else:
                                        self.insert_gate(qfi_circuit,
                                                         gates_to_parameters[target_params[0]][0],
                                                         ZGate(),
                                                         qubits=[work_qubit])
                                else:
                                    if is_complex:
                                        self.insert_gate(qfi_circuit,
                                                         gates_to_parameters[target_params[0]][0],
                                                         SGate(),
                                                         qubits=[work_qubit])

                                self.insert_gate(qfi_circuit,
                                                 gates_to_parameters[target_params[0]][0],
                                                 XGate(),
                                                 qubits=[work_qubit])

                                # Insert controlled, intercepting gate - controlled by |1>
                                self.insert_gate(qfi_circuit,
                                                 gates_to_parameters[param_i][m_i],
                                                 gate_to_insert_i,
                                                 additional_qubits=additional_qubits)

                                self.insert_gate(qfi_circuit,
                                                 gate_to_insert_i,
                                                 XGate(),
                                                 qubits=[work_qubit],
                                                 after=True)

                                # Insert controlled, intercepting gate - controlled by |0>
                                self.insert_gate(qfi_circuit,
                                                 gates_to_parameters[param_j][m_j],
                                                 gate_to_insert_j,
                                                 additional_qubits=additional_qubits)

                                # Remove redundant gates

                                if j <= i:
                                    qfi_circuit = self.trim_circuit(
                                        qfi_circuit, gates_to_parameters[param_i][m_i]
                                        )
                                else:
                                    qfi_circuit = self.trim_circuit(
                                        qfi_circuit, gates_to_parameters[param_j][m_j]
                                        )

                                qfi_circuit.h(work_qubit)
                                # Convert the quantum circuit into a CircuitStateFn
                                term = np.sqrt(np.abs(coeff_i) * np.abs(coeff_j)) * op.coeff * \
                                    CircuitStateFn(qfi_circuit)

                                # Chain Rule Parameter Expression

                                gate_param_i = gates_to_parameters[param_i][m_i].params[k_i]
                                gate_param_j = gates_to_parameters[param_j][m_j].params[k_j]

                                meas = deepcopy(qfi_observable)
                                if isinstance(gate_param_i, ParameterExpression):
                                    import sympy as sy
                                    expr_grad = self.parameter_expression_grad(gate_param_i, param_i)
                                    meas *= expr_grad
                                if isinstance(gate_param_j, ParameterExpression):
                                    import sympy as sy
                                    expr_grad = self.parameter_expression_grad(gate_param_j, param_j)
                                    meas *= expr_grad
                                term = meas @ term

                                if m_i == 0 and k_i == 0 and m_j == 0 and k_j == 0:
                                    qfi_op = term
                                else:
                                    # Product Rule
                                    qfi_op += term
                # Compute −4 * Re(〈∂kψ|ψ〉〈ψ|∂lψ〉)
                def phase_fix_combo_fn(x):
                    return 4*(-0.5)*(x[0]*np.conjugate(x[1]) + x[1]*np.conjugate(x[0]))
                phase_fix = ListOp([phase_fix_states[i], phase_fix_states[j]],
                                   combo_fn=phase_fix_combo_fn)
                # Add the phase fix quantities to the entries of the QFI
                # Get 4 * Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉]
                qfi_ops += [qfi_op + phase_fix]
            qfi_operators.append(ListOp(qfi_ops))
        # Return the full QFI
        return ListOp(qfi_operators)

    def block_diagonal_qfi(self,
                           operator: Union[CircuitOp, CircuitStateFn],
                           params: Optional[Union[Parameter,
                                                          ParameterVector,
                                                          List[Parameter]]] = None
                           ) -> OperatorBase:

        """TODO"""
        if not isinstance(operator, (CircuitOp, CircuitStateFn)):
            raise NotImplementedError('operator must be a CircuitOp or CircuitStateFn')

        circuit = operator.primitive

        # Parition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = self._partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        block_params = [list(layer.parameters) for layer in layers]
        # Remove any parameters found which are not in params
        block_params = [[param for param in block if param in params] for block in block_params]
        
        # Determine the permutation needed to ensure that the final
        # operator is consistent with the ordering of the input parameters
        perm = [params.index(param) for block in block_params for param in block]
        
        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # Get generators
        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.
        # TODO: currently the input: target_params is ignored. This should either be removed,
        # or logic should be added to prevent evaluating the QFI on certain params.

        generators = self.get_generators(params, circuit)
        #param_expressions = 

        blocks = []

        #Psi_i = layer_i @ layer_i-1 @ ... @ layer_0 @ Zero
        for k, psi_i in enumerate(psis):
            params = block_params[k]
            block = np.zeros((len(params), len(params))).tolist()

            # calculate all single-operator terms <psi_i|generator_i|psi_i>
            single_terms = np.zeros(len(params)).tolist()
            for i, p_i in enumerate(params):
                generator = generators[p_i]
                psi_gen_i = ~StateFn(generator) @ psi_i @ Zero
                psi_gen_i = PauliExpectation().convert(psi_gen_i)
                single_terms[i] = psi_gen_i

            def get_parameter_expression(circuit, param):
                if len(circuit._parameter_table[param]) > 1:
                    raise NotImplementedError("The QFI Approximations do not yet support multiple "
                                             "gates parameterized by a single parameter. For such circuits "
                                             "set approx = None")
                gate = circuit._parameter_table[param][0][0]
                assert len(gate.params) == 1, "Circuit was not properly decomposed"
                param_value = gate.params[0]
                return param_value

            # Calculate all double-operator terms <psi_i|generator_j @ generator_i|psi_i>
            # and build composite operators for each matrix entry
            for i, p_i in enumerate(params):
                generator_i = generators[p_i]
                param_expr_i = get_parameter_expression(circuit, p_i)

                for j, p_j in enumerate(params):
                    if i == j:
                        block[i][i] = ListOp([single_terms[i]], combo_fn=lambda x: 1 - x[0] ** 2)
                        if isinstance(param_expr_i, ParameterExpression) and not isinstance(param_expr_i, Parameter):
                            expr_grad_i = self.parameter_expression_grad(param_expr_i, p_i)
                            block[i][j] *= (expr_grad_i)*(expr_grad_i)
                        continue

                    generator_j = generators[p_j]
                    generator = ~generator_j @ generator_i
                    param_expr_j = get_parameter_expression(circuit, p_j)

                    

                    psi_gen_ij = ~StateFn(generator) @ psi_i @ Zero
                    psi_gen_ij = PauliExpectation().convert(psi_gen_ij)
                    cross_term = ListOp([single_terms[i], single_terms[j]], combo_fn=np.prod)
                    block[i][j] = psi_gen_ij - cross_term

                    if isinstance(param_expr_i, ParameterExpression) and not isinstance(param_expr_i, Parameter):
                        expr_grad_i = self.parameter_expression_grad(param_expr_i, p_i)
                        block[i][j] *= expr_grad_i
                    if isinstance(param_expr_j, ParameterExpression) and not isinstance(param_expr_j, Parameter):
                        expr_grad_j = self.parameter_expression_grad(param_expr_j, p_j)
                        block[i][j] *= expr_grad_j

            wrapped_block = ListOp([ListOp(row) for row in block])
            blocks.append(wrapped_block)

        block_diagonal_qfi = ListOp(oplist=blocks, combo_fn=lambda x: np.real(block_diag(*x))[:,perm][perm,:])
        return block_diagonal_qfi

    def diagonal_qfi(self,
                     operator: Union[CircuitOp, CircuitStateFn],
                     params: Union[Parameter, ParameterVector, List] = None
                     ) -> OperatorBase:

        """TODO"""
        if not isinstance(operator, (CircuitOp, CircuitStateFn)):
            raise NotImplementedError

        circuit = operator.primitive

        # Parition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = self._partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.
        generators = self.get_generators(params, circuit)

        diag = []
        for param in params:
            if len(circuit._parameter_table[param]) > 1:
                raise NotImplementedError("The QFI Approximations do not yet support multiple "
                                         "gates parameterized by a single parameter. For such circuits "
                                         "set approx = None")
                
            gate = circuit._parameter_table[param][0][0]
            
            assert len(gate.params) == 1, "Circuit was not properly decomposed"
            
            param_value = gate.params[0]
            generator = generators[param]
            meas_op = ~StateFn(generator)

            # get appropriate psi_i
            psi = [(psi) for psi in psis if param in psi.primitive.parameters][0]

            op = meas_op @ psi @ Zero
            if isinstance(param_value, ParameterExpression) and not isinstance(param_value, Parameter):
                            expr_grad = self.parameter_expression_grad(param_value, param)
                            op *= expr_grad
            rotated_op = PauliExpectation().convert(op)
            diag.append(rotated_op)

        grad_op = ListOp(diag, combo_fn=lambda x: np.diag(np.real([1 - y ** 2 for y in x])))
        return grad_op

    def _partition_circuit(self, circuit):
        """TODO"""
        dag = circuit_to_dag(circuit)
        dag_layers = ([i['graph'] for i in dag.serial_layers()])
        num_qubits = circuit.num_qubits
        layers = list(
            zip(dag_layers, [{x: False for x in range(0, num_qubits)} for layer in dag_layers]))

        # initialize the ledger
        # The ledger tracks which qubits in each layer are available to have
        # gates from subsequent layers shifted backward.
        # The idea being that all parameterized gates should have
        # no descendants within their layer
        for i, (layer, ledger) in enumerate(layers):
            op_node = layer.op_nodes()[0]
            is_param = op_node.op.is_parameterized()
            qargs = op_node.qargs
            indices = [qarg.index for qarg in qargs]
            if is_param:
                for index in indices:
                    ledger[index] = True

        def apply_node_op(node, dag, back=True):
            op = copy.copy(node.op)
            qargs = copy.copy(node.qargs)
            cargs = copy.copy(node.cargs)
            condition = copy.copy(node.condition)
            if back:
                dag.apply_operation_back(op, qargs, cargs, condition)
            else:
                dag.apply_operation_front(op, qargs, cargs, condition)

        converged = False

        for _ in range(dag.depth()+1):
            if converged:
                break

            converged = True

            for i, (layer, ledger) in enumerate(layers):
                if i == len(layers) - 1:
                    continue

                (next_layer, next_ledger) = layers[i + 1]
                for next_node in next_layer.op_nodes():
                    is_param = next_node.op.is_parameterized()
                    qargs = next_node.qargs
                    indices = [qarg.index for qarg in qargs]

                    # If the next_node can be moved back a layer without
                    # without becoming the descendant of a parameterized gate, 
                    # then do it.
                    if not any([ledger[x] for x in indices]):

                        apply_node_op(next_node, layer)
                        next_layer.remove_op_node(next_node)

                        if is_param:
                            for index in indices:
                                ledger[index] = True
                                next_ledger[index] = False

                        converged = False

                # clean up empty layers left behind.
                if len(next_layer.op_nodes()) == 0:
                    layers.pop(i+1)

        partitioned_circs = [dag_to_circuit(layer[0]) for layer in layers]
        return partitioned_circs

    def get_generators(self, params, circuit):
        """TODO"""
        dag = circuit_to_dag(circuit)
        layers = list(dag.serial_layers())

        generators = {}
        num_qubits = dag.num_qubits()

        for layer in layers:
            instr = layer['graph'].op_nodes()[0].op
            if len(instr.params) == 0:
                continue
            assert len(instr.params) == 1, "Circuit was not properly decomposed"
            param_value = instr.params[0]
            for param in params:
                if param in param_value.parameters:

                    if isinstance(instr, RYGate):
                        generator = Y
                    elif isinstance(instr, RZGate):
                        generator = Z
                    elif isinstance(instr, RXGate):
                        generator = X
                    else:
                        raise NotImplementedError

                    # get all qubit indices in this layer where the param parameterizes
                    # an operation.
                    indices = [[q.index for q in qreg] for qreg in layer['partition']]
                    indices = [item for sublist in indices for item in sublist]

                    if len(indices) > 1:
                        raise NotImplementedError
                    index = indices[0]
                    generator = (I ^ (index)) ^ generator ^ (I ^ (num_qubits-index-1))
                    generators[param] = generator

        return generators

    def _sort_params(self, params):
        def compare_params(param1, param2):
            name1 = param1.name
            name2 = param2.name
            value1 = name1[name1.find("[")+1:name1.find("]")]
            value2 = name2[name2.find("[")+1:name2.find("]")]
            return int(value1) - int(value2)
        return sorted(params, key=cmp_to_key(compare_params), reverse=False)
