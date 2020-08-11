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

from typing import Optional, Tuple, List, Dict, Union, Callable
import warnings

import numpy as np
import copy
from copy import deepcopy
from functools import partial, reduce, cmp_to_key

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp, PauliOp
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from ...expectations import PauliExpectation 
from qiskit.aqua.operators.converters import DictToCircuitSum
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.aqua.operators.operator_globals import H, S, I, Z, Y, Zero, One


from qiskit.extensions.standard import HGate, XGate, SdgGate, SGate, ZGate

from qiskit.aqua import QuantumInstance
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.library import RYGate, RZGate, RXGate
from qiskit.dagcircuit  import DAGCircuit
from qiskit.quantum_info import Pauli

from ..gradient_base import GradientBase



class QFI(GradientBase):
    r"""Compute the Quantum Fisher Information given a pure, parametrized quantum state.
        [QFI]kl= Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉] * 0.25.
    """

    def convert(self,
                operator: OperatorBase = None,
                params: Union[Parameter, ParameterVector, List] = None) -> OperatorBase:
        r"""
        Args
            operator:The operator corresponding to our quantum state we are taking the gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
        Returns
            ListOp[ListOp] where the operator at position k,l corresponds to [QFI]kl
        """
        # TODO integrate diagonal without ancilla

        return self._prepare_operator(operator)

    def _prepare_operator(self, operator, params):
        if isinstance(operator, ListOp):
            return operator.traverse(self.prepare_operator)
        elif isinstance(operator, StateFn):
            if operator.is_measurement == True:
                return operator.traverse(self.prepare_operator)
        elif isinstance(operator, PrimitiveOp):
            return 4 * ((I ^ operator.num_qubits) ^ Z - operator ^ Z)  # Z needs to be at the end
        if isinstance(operator, (QuantumCircuit, CircuitStateFn, CircuitOp)):
            # operator.primitive.add_register(QuantumRegister(1, name="ancilla"))
            operator = self._qfi_states(operator, params)
        return operator

    def _qfi_states(self, op: OperatorBase,
                    target_params: Union[Parameter, ParameterVector, List] = None) -> ListOp:
        """Generate the operators whose evaluation leads to the full QFI.

        Args:
            op: The operator representing the quantum state for which we compute the QFI.
            target_params: The parameters we are computing the QFI wrt: ω

        Returns:
            Operators which give the QFI.
            If a parameter appears multiple times, one circuit is created per parameterized gates to compute
            the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
        """
        # QFI & phase fix observable
        # qfi_observable = 4 * ((I ^ op.num_qubits) ^ Z - op ^ Z)
        # phase_fix_observable = (I ^ op.num_qubits) ^ (X + 1j * Y)  # see https://arxiv.org/pdf/quant-ph/0108146.pdf
        # Dictionary with the information which parameter is used in which gate
        gates_to_parameters = {}
        # Dictionary which relates the coefficients needed for the QFI for every parameter
        qfi_coeffs = {}
        # Dictionary which relates the gates needed for the QFI for every parameter
        qfi_gates = {}
        # Loop throuh the parameters in the circuit
        params = []

        if isinstance(op, CircuitStateFn) or isinstance(op, CircuitOp):
            pass
        elif isinstance(op, DictStateFn) or isinstance(op, VectorStateFn):
            op = DictToCircuitSum.convert(op)
        else:
            raise TypeError('Ancilla gradients only support operators whose states are either '
                            'CircuitStateFn, DictStateFn, or VectorStateFn.')
        state_qc = deepcopy(op.primitive)
        for param, elements in state_qc._parameter_table.items():
            if param not in target_params:
                continue
            params.append(param)
            gates_to_parameters[param] = []
            qfi_coeffs[param] = []
            qfi_gates[param] = []
            for element in elements:
                # get the coefficients and controlled gates (raises an error if the parameterized gate is not supported)
                coeffs_gates = self.gate_gradient_dict(element[0])
                gates_to_parameters[param].append(element[0])
                for c_g in coeffs_gates:
                    qfi_coeffs[param].append(c_g[0])
                    qfi_gates[param].append(c_g[1])

        qfi_operators = []
        qr_ancilla = QuantumRegister(1, 'ancilla')
        ancilla = qr_ancilla[0]
        additional_qubits = ([ancilla], [])
        # create a copy of the original circuit with an additional ancilla register
        circuit = QuantumCircuit(*state_qc.qregs, qr_ancilla)
        circuit.data = state_qc.data
        params = list(gates_to_parameters.keys())
        # apply Hadamard on ancilla
        self.insert_gate(circuit, gates_to_parameters[params[0]][0], HGate(),
                    qubits=[ancilla])
        # Get the circuits needed to compute A_ij
        for i in range(len(params)): #loop over parameters
            qfi_ops = []
            # TODO Check if this overhead can be reduced or is cached by/with the OpFlow
            # j = 0
            # while j <= i: #loop over parameters
            for j in range(len(params)):

                # construct the circuits
                for m, gates_to_insert_i in enumerate(qfi_gates[params[i]]):
                    for k, gate_to_insert_i in enumerate(gates_to_insert_i):
                        coeff_i = qfi_coeffs[params[i]][m][k]
                        for n, gates_to_insert_j in enumerate(qfi_gates[params[j]]):
                            for l, gate_to_insert_j in enumerate(gates_to_insert_j):
                                coeff_j = qfi_coeffs[params[j]][n][l]
                                # create a copy of the original circuit with the same registers
                                qfi_circuit = QuantumCircuit(*circuit.qregs)
                                qfi_circuit.data = circuit.data

                                # Fix ancilla phase
                                sign = np.sign(np.conj(coeff_i)*coeff_j)
                                complex = np.iscomplex(np.conj(coeff_i)*coeff_j)
                                if sign == -1:
                                    if complex:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], SdgGate(),
                                                    qubits=[ancilla])
                                    else:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], ZGate(),
                                                    qubits=[ancilla])
                                else:
                                    if complex:
                                        self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], SGate(),
                                                    qubits=[ancilla])

                                self.insert_gate(qfi_circuit, gates_to_parameters[params[0]][0], XGate(),
                                            qubits=[ancilla])

                                # Insert controlled, intercepting gate - controlled by |1>
                                self.insert_gate(qfi_circuit, gates_to_parameters[params[i]][m], gate_to_insert_i,
                                                                         additional_qubits=additional_qubits)

                                self.insert_gate(qfi_circuit, gate_to_insert_i, XGate(), qubits=[ancilla], after=True)

                                # Insert controlled, intercepting gate - controlled by |0>
                                self.insert_gate(qfi_circuit, gates_to_parameters[params[j]][n], gate_to_insert_j,
                                                                         additional_qubits=additional_qubits)

                                '''TODO check if we could use the trimming 
                                What speaks against it is the new way to compute the phase fix directly within 
                                the observable here the trimming wouldn't work. The other way would be more efficient
                                in terms of computation but this is more convenient to write it.'''

                                # Remove redundant gates
                                # qfi_circuit = self.trim_circuit(qfi_circuit, gates_to_parameters[params[i]][m])

                                qfi_circuit.h(ancilla)
                                if m == 0 and k == 0:
                                    qfi_op = [np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(qfi_circuit)]
                                else:
                                    qfi_op += np.abs(coeff_i) * np.abs(coeff_j) * CircuitStateFn(qfi_circuit)
                qfi_ops += [qfi_op]
            qfi_operators.append(qfi_ops)
        return ListOp(qfi_operators)

    def old_convert(self, operator: OperatorBase,
                approximation: str = 'diagonal',
                method: str = 'no_ancilla') -> OperatorBase:
        r"""
        Args:
            operator: The operator for which we are generating the Quantum Fisher Information Metric Tensor
            parameters: The parameters over which the metric tensor is being evaluated

        Returns:
            metric_operator: An operator whose evaluation yeilds the QFI metric tensor
        """

        if approximation == 'block_diagonal' and method == 'no_ancilla':

            if not isinstance(operator, (CircuitOp,CircuitStateFn)):
                raise NotImplementedError

            circuit = operator.primitive

            #Parition the circuit into layers, and build the circuits to prepare $\psi_l$
            layers = self.partition_circuit(circuit)
            if layers[-1].num_parameters == 0:
                layers.pop(-1)
                
            psis = [CircuitOp(layer) for layer in layers]
            for i, psi in enumerate(psis):
                if i == 0:  continue
                psis[i] = psi@psis[i-1]

            #Get generators
            #TODO: make this work for other types of rotations
            #NOTE: This assumes that each parameter only affects one rotation.
            # we need to think more about what happens if multiple rotations 
            # are controlled with a single parameter. 
            params = circuit.ordered_parameters
            generators = self.get_generators(params,circuit)

            blocks = []

            for l, psi_l in enumerate(psis):

                params = self.sort_params(psi_l.primitive.parameters)

                block = np.zeros((len(params),len(params))).tolist()
                

                #calculate all single-operator terms <psi_l|K_i|psi_l>
                single_terms = np.zeros(len(params)).tolist()
                for i, pi in enumerate(params):
                    K = generators[pi]
                    psi_Ki = ~StateFn(K) @ psi_l @ Zero
                    psi_Ki = PauliExpectation().convert(psi_Ki)
                    single_terms[i] = psi_Ki


                
                #Calculate all double-operator terms <psi_l|K_j @ K_i|psi_l>
                # and build composite operators for each matrix entry
                for i, pi in enumerate(params):
                    Ki = generators[pi]
                    for j, pj in enumerate(params):

                        if i == j:
                            block[i][i] = ListOp(oplist=[single_terms[i]], combo_fn=lambda x: [1-y**2 for y in x])
                            continue

                        Kj = generators[pj]
                        K = ~Kj @ Ki

                        psi_KjKi = ~StateFn(K) @ psi_l @ Zero
                        psi_KjKi = PauliExpectation().convert(psi_KjKi)
                        cross_term = ListOp(oplist = [single_terms[i],single_terms[j]], combo_fn= lambda x: np.prod(x))
                        block[i][j] = psi_KjKi - cross_term

                blocks.append(block)

            qfi = blocks[0]
            if len(blocks) > 1:
                for block in blocks[1:]:
                    qfi = block_diag(qfi, block)

            #for i, row in enumerate(qfi):
            #    qfi[i] = ListOp(row)
            #qfi = ListOp(qfi)

            return qfi

        # layers = partition_circuit()
        elif approximation == 'diagonal' and method == 'no_ancilla':

            if not isinstance(operator, (CircuitOp,CircuitStateFn)):
                raise NotImplementedError

            circuit = operator.primitive

            #Parition the circuit into layers, and build the circuits to prepare $\psi_l$
            layers = self.partition_circuit(circuit)
            if layers[-1].num_parameters == 0:
                layers.pop(-1)
                
            psis = [CircuitOp(layer) for layer in layers]
            for i, psi in enumerate(psis):
                if i == 0:  continue
                psis[i] = psi@psis[i-1]

            #Get generators
            #TODO: make this work for other types of rotations
            #NOTE: This assumes that each parameter only affects one rotation.
            # we need to think more about what happens if multiple rotations 
            # are controlled with a single parameter. 
            params = circuit.ordered_parameters
            generators = self.get_generators(params,circuit)

            diag = []
            for param in params:
                K = generators[param]
                meas_op = ~StateFn(K)
                
                #get appropriate psi_l
                psi = [(psi) for psi in psis if param in psi.primitive.parameters][0]
                
                op = meas_op @ psi @ Zero
                rotated_op = PauliExpectation().convert(op)
                diag.append(rotated_op)
                
            grad_op = ListOp(oplist=diag, combo_fn=lambda x: [1-y**2 for y in x])
            return grad_op



        return grad_op

    def partition_circuit(self, circuit):
        dag = circuit_to_dag(circuit)
        dag_layers = ([i['graph'] for i in dag.serial_layers()])
        num_qubits = circuit.num_qubits
        layers = list(zip(dag_layers, [{x:False for x in range(0,num_qubits)} for layer in dag_layers]))

        #initialize the ledger
        # The ledger tracks which qubits in each layer are available to have 
        # gates from subsequent layers shifted backward.
        # The idea being that all parameterized gates should have 
        # no descendants within their layer
        for i,(layer,ledger) in enumerate(layers):
            op_node = layer.op_nodes()[0]
            is_param = op_node.op.is_parameterized()
            qargs = op_node.qargs
            indices = [qarg.index for qarg in qargs]   
            if is_param:
                for index in indices:
                    ledger[index] = True

        
        def apply_node_op(node, dag, back=True):
            op = copy.copy(node.op)
            qa = copy.copy(node.qargs)
            ca = copy.copy(node.cargs)
            co = copy.copy(node.condition)
            if back:
                dag.apply_operation_back(op, qa, ca, co)
            else:
                dag.apply_operation_front(op, qa, ca, co)

        converged = False
        
        
        
        for x in range(dag.depth()):  
            if converged: 
                break

            converged = True

            for i,(layer,ledger) in enumerate(layers):
                if i == len(layers)-1: continue

                (next_layer, next_ledger) = layers[i+1]
                for next_node in next_layer.op_nodes():
                    is_param = next_node.op.is_parameterized()
                    qargs = next_node.qargs
                    indices = [qarg.index for qarg in qargs]   

                    #If the next_node can be moved back a layer without conflicting 
                    # without becoming the descendant of a parameterized gate, then do it.
                    if not any([ledger[x] for x in indices]):

                        apply_node_op(next_node, layer)
                        next_layer.remove_op_node(next_node)

                        if is_param:
                            for index in indices:
                                ledger[index] = True
                                next_ledger[index] = False

                        converged = False

                #clean up empty layers left behind.
                if len(next_layer.op_nodes()) == 0:
                    layers.pop(i+1)

        partitioned_circs = [dag_to_circuit(layer[0]) for layer in layers]
        return partitioned_circs

    def get_generators(self, params, circuit):
        dag = circuit_to_dag(circuit)
        layers = list(dag.serial_layers())

        generators = {}
        num_qubits = dag.num_qubits()

        for layer in layers:
            instr = layer['graph'].op_nodes()[0].op
            for param in params:
                if param in instr.params:

                    if isinstance(instr, RYGate):
                        K = Y
                    elif isinstance(instr, RZGate):
                        K = Z
                    elif isinstance(instr, RXGate):
                        K = X
                    else:
                        raise NotImplementedError 

                    #get all qubit indices in this layer where the param parameterizes
                    # an operation.
                    indices = [[q.index for q in qreg] for qreg in layer['partition']]
                    indices = [item for sublist in indices for item in sublist]

                    if len(indices) > 1:
                        raise NoteImplementedError
                    index = indices[0]
                    K = (I^(index))^K^(I^(num_qubits-index-1))
                    generators[param] = K

        return generators

    def sort_params(self, params):
        def compare_params(p1, p2):
            s1 = p1.name
            s2 = p2.name
            v1= s1[s1.find("[")+1:s1.find("]")]
            v2= s2[s2.find("[")+1:s2.find("]")]
            return int(v1) - int(v2)
        return sorted(params, key=cmp_to_key(compare_params),reverse=False)
