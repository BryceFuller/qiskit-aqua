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

""" GradientBase Class """

from collections.abc import Iterable

from typing import Optional, Callable, Union, List, Tuple
import logging
from functools import partial, reduce
import numpy as np
import sympy as sy
from copy import deepcopy

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library.standard_gates import *
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit import Gate, Instruction, Qubit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from ..converters import CircuitSampler

from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..primitive_ops.pauli_op import PauliOp
from ..primitive_ops.circuit_op import CircuitOp
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.tensored_op import TensoredOp
from ..state_fns.state_fn import StateFn
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..operator_globals import H, S, I, Zero, One
from ..converters.converter_base import ConverterBase

logger = logging.getLogger(__name__)


class GradientBase(ConverterBase):
    r"""
    Converter for changing parameterized circuits into operators
    whose evaluation yields the gradient with respect to the circuit parameters.
    """
    # Todo remove
    """
    def decompose_to_two_unique_eigenval(self,
                                        operator: OperatorBase,
                                        params: Union[Parameter, ParameterVector, List])-> OperatorBase:

        Decompose the input circuit so that all gates which will be differentiated
        have two unique eigenvalues
        Args:
            state_operator: the state prep circuit we wish to decompose
        Returns:
            state_operator: An equivalent quantum circuit such that all (relevant)
            parameterized gates are decomposed into gates with two unique eigenvalues.


        return OperatorBase
    """
    # TODO discuss naming
    def get_callable(self,
                      operator: OperatorBase,
                      params: Union[Parameter, ParameterVector, List[Parameter]],
                      backend: Optional[Union[BaseBackend, QuantumInstance]] = None) -> callable:
        """
        Get a callable function which provides the respective gradient, Hessian or QFI for given parameter values.
        This callable can be used as gradient function for optimizers.
        Args:
            operator: The operator for which we want to get the gradient, Hessian or QFI.
            parameters: The parameters with respect to which we are taking the gradient, Hessian or QFI.
            backend: The quantum backend or QuantumInstance to use to evaluate the gradient, Hessian or QFI.
        Returns:
            callable: Function to compute a gradient, Hessian or QFI for given parameters.

        """

        if not backend:
            converter = self.convert(operator, params)
        else:
            if isinstance(backend, QuantumInstance):
                if backend.is_statevector:
                    converter = self.convert(operator, params)
                else:
                    converter = CircuitSampler(backend=backend).convert(self.convert(operator, params))
            else:
                if backend.name().startswith('statevector'):
                    converter = self.convert(operator, params)
                else:
                    converter = CircuitSampler(backend=backend).convert(self.convert(operator, params))
        return lambda p_values: converter.bind_params(dict(zip(params, p_values))).eval()


    def _get_gate_generator(self, operator, param):
        r"""
        Need to figure out what gate param corresponds to and return the
        Hermitian generator of that rotation.
        """

        return

    #TODO I need to think about how this method will traverse operator,
    #  for example, if operator contains multiple different circuits,
    #  then we may need to worry about name collisions!

    # def parameter_expression_grad(self, pe, param):
    #     deriv =sy.diff(sy.sympify(str(pe)), param)
    #
    #     symbol_map = {}
    #     symbols = deriv.free_symbols
    #
    #     for s in symbols:
    #         for p in pe.parameters:
    #             if s.name == p.name:
    #                 symbol_map[p] = s
    #                 break
    #     assert len(symbols) == len(symbol_map), "Unaccounted for symbols!"
    #
    #     return ParameterExpression(symbol_map, deriv)


    def gate_gradient_dict(self,
                           gate: Gate) -> List[Tuple[List[complex], List[Instruction]]]:

        """Given a parameterized gate U(theta) with derivative dU(theta)/dtheta = sum_ia_iU(theta)V_i.
           This function returns a:=[a_0, ...] and V=[V_0, ...]
           Suppose U takes multiple parameters, i.e., U(theta^0, ... theta^k).
           The returned coefficients and gates are ordered accordingly.
           Only parameterized Qiskit gates are supported.

           Args:
                gate: The gate for which the derivative is being computed.

           Returns:
                The coefficients and the gates used for the metric computation for each parameter of the respective gates.
                [([a^0], [V^0]) ..., ([a^k], [V^k])]

           Raises:
                TypeError: If the input gate is not a supported parametrized gate."""

        if isinstance(gate, U1Gate):
            # theta
            return [([0.5j, -0.5j], [IGate(), CZGate()])]
        if isinstance(gate, U2Gate): #Going to be deprecated
            # theta, phi
            return [([-0.5j], [CZGate()]), ([0.5j], [CZGate()])]
        if isinstance(gate, U3Gate):
            # theta, lambda, phi
            return [([-0.5j], [CZGate()]), ([+0.5j], [CZGate()]), ([-0.5j], [CZGate()])]
        if isinstance(gate, RXGate):
            # theta
            return [([-0.5j], [CXGate()])]
        if isinstance(gate, RYGate):
            # theta
            return [([-0.5j], [CYGate()])]
        if isinstance(gate, RZGate):
            # theta
            return [([-0.5j], [CZGate()])]
        if isinstance(gate, RXXGate):
            # theta
            cxx_circ = QuantumCircuit(3)
            cxx_circ.cx(0, 1)
            cxx_circ.cx(0, 2)
            cxx = cxx_circ.to_instruction()
            return [([-0.5j], [cxx])]
        if isinstance(gate, RYYGate):
            # theta
            cyy_circ = QuantumCircuit(3)
            cyy_circ.cy(0, 1)
            cyy_circ.cy(0, 2)
            cyy = cyy_circ.to_instruction()
            return [([-0.5j], [cyy])]
        if isinstance(gate, RZZGate):
            # theta
            czz_circ = QuantumCircuit(3)
            czz_circ.cz(0, 1)
            czz_circ.cz(0, 2)
            czz = czz_circ.to_instruction()
            return [([-0.5j], [czz])]
        if isinstance(gate, RZXGate):
            # theta
            czx_circ = QuantumCircuit(3)
            czx_circ.cx(0, 1)
            czx_circ.cz(0, 2)
            czx = czx_circ.to_instruction()
            return [([-0.5j], [czx])]
        if isinstance(gate, ControlledGate):
            # TODO support arbitrary control states
            if gate.ctrl_state != 2**gate.num_ctrl_qubits - 1:
                raise AquaError('Function only support controlled gates with control state `1` on all control qubits.')

            base_coeffs_gates = self.gate_gradient_dict(gate.base_gate)
            coeffs_gates = []
            # The projectors needed for the gradient of a controlled gate are integrated by a sum of gates.
            # The following line generates the decomposition gates.
            proj_gates_controlled = [[(-1) ** p.count(ZGate()), p] for p in product([IGate(), ZGate()],
            repeat=gate.num_ctrl_qubits)]
            for base_coeffs, base_gates in base_coeffs_gates: # loop over parameters
                coeffs = [c / (2 ** gate.num_ctrl_qubits) for c in base_coeffs]
                gates = []
                for phase, proj_gates in proj_gates_controlled:
                    base_coeffs.extend(phase * c for c in base_coeffs)
                    for base_gate in base_gates:
                        controlled_circ = QuantumCircuit(gate.num_ctrl_qubits + gate.num_qubits + 1)
                        for i, proj_gate in enumerate(proj_gates):
                            if isinstance(proj_gate, ZGate):
                                controlled_circ.cz(0, i)
                        if not isinstance(base_gate, IGate):
                            controlled_circ.append(base_gate, [0, range(gate.num_ctrl_qubits + 1, gate.num_ctrl_qubits +
                                                                    gate.num_qubits + 1)])
                        # TODO make sure that the list before does the right thing. i.e. append a controlled gate to the
                        # TODO ancilla 0 and the gates considering the base_gate
                        gates.append(controlled_circ.to_instruction())
                coeffs_gates.append((coeffs, gates))
            return coeffs_gates

        raise TypeError('Unrecognized parametrized gate, {}'.format(gate))

    # TODO get ParameterExpression in the different gradients
    def parameter_expression_grad(self,
                                  param_expr: ParameterExpression,
                                  param: Parameter) -> ParameterExpression:

        """Get the derivative of a parameter expression w.r.t. the given parameter.

        Args:
            param_expr: The Parameter Expression for which we compute the derivative
            param: Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
        """
        deriv =sy.diff(sy.sympify(str(param_expr)), str(param))
        
        symbol_map = {}
        symbols = deriv.free_symbols
        
        for s in symbols:
            for p in param_expr.parameters:
                if s.name == p.name:
                    symbol_map[p] = s
                    break

        assert len(symbols) == len(symbol_map), "Unaccounted for symbols!"
        
        return ParameterExpression(symbol_map, deriv)


        """#I don't understand how this function works or what exactly it's trying to do
        # I'm not deleting it, but I need to use the old implementation for now so my code doesn't break.
        expr = param_expr._symbol_expr
        keys = param._parameter_symbols[param]
        expr_grad = 0
        for key in keys:
            expr_grad += sy.Derivative(expr, key)
        return ParameterExpression(param_expr._parameter_symbols, expr = expr_grad)
        #"""

    def insert_gate(self,
                    circuit: QuantumCircuit,
                    reference_gate: Gate,
                    gate_to_insert: Instruction,
                    qubits: Optional[List[Qubit]] = None,
                    additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None,
                    after: bool = False):

        """Insert a gate into the circuit.

        Args:
            circuit: The circuit onto which the gare is added.
            reference_gate: A gate instance before or after which a gate is inserted.
            gate_to_insert: The gate to be inserted.
            qubits: The qubits on which the gate is inserted. If None, the qubits of the
                reference_gate are used.
            additional_qubits: If qubits is None and the qubits of the reference_gate are
                used, this can be used to specify additional qubits before (first list in
                tuple) or after (second list in tuple) the qubits.
            after: If the gate_to_insert should be inserted after the reference_gate set True.
        """

        if isinstance(gate_to_insert, IGate):
            return
        else:
            for i, op in enumerate(circuit.data):
                if op[0] == reference_gate:
                    qubits = qubits or op[1]
                    if additional_qubits:
                        qubits = additional_qubits[0] + qubits + additional_qubits[1]
                    op_to_insert = (gate_to_insert, qubits, [])
                    if after:
                        insertion_index = i+1
                    else:
                        insertion_index = i
                    circuit.data.insert(insertion_index, op_to_insert)
                    return
            raise AquaError('Could not insert the controlled gate, something went wrong!')

    def trim_circuit(self,
                     circuit: QuantumCircuit,
                     reference_gate: Gate) -> QuantumCircuit:

        """Trim the given quantum circuit before the reference gate.

        Args:
            circuit: The circuit onto which the gare is added.
            reference_gate: A gate instance before or after which a gate is inserted.

        Returns:
            The trimmed circuit.

        Raise:
            AquaError: If the reference gate is not part of the given circuit.
        """
        parameterized_gates = []
        for _, elements in circuit._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                trimmed_circuit = QuantumCircuit(*circuit.qregs)
                trimmed_circuit.data = circuit.data[:i]
                return trimmed_circuit

        raise AquaError('The reference gate is not in the given quantum circuit.')

    @classmethod
    def unroll_operator(cls, operator: OperatorBase) -> Union[OperatorBase, List[OperatorBase]]:
        if isinstance(operator, ListOp):
            return [cls.unroll_operator(op) for op in operator]
        if hasattr(operator, 'primitive') and isinstance(operator.primitive, ListOp):
            return [operator.__class__(op) for op in operator.primitive]
        return operator

    @classmethod
    def get_unique_circuits(cls, operator: OperatorBase) -> List[QuantumCircuit]:
        def get_circuit(op):
            if isinstance(op, (CircuitStateFn, CircuitOp)):
                return op.primitive
        
        unrolled_op = cls.unroll_operator(operator)
        circuits = []
        for ops in unrolled_op:
            if not isinstance(ops, list):
                ops = [ops]
            for op in ops:
                if isinstance(op, (CircuitStateFn, CircuitOp, QuantumCircuit)):
                    c = get_circuit(op)
                    if c not in circuits:
                        circuits.append(c)
        return circuits

    def append_Z_measurement(self, operator):
        if isinstance(operator, ListOp):
            return operator.traverse(self.append_Z_measurement)
        elif isinstance(operator,StateFn):
            if operator.is_measurement == True:
                return operator.traverse(self.append_Z_measurement)      
        elif isinstance(operator, PauliOp):
            return (Z^operator)
        if isinstance(operator,(QuantumCircuit,CircuitStateFn, CircuitOp)):
            #print((operator))
            
            operator.primitive.add_register(QuantumRegister(1, name="ancilla"))   
        
        return operator

    # For now not needed
    # @staticmethod
    # def replace_gate(circuit: QuantumCircuit,
    #                 gate_to_replace: Gate,
    #                 gate_to_insert: Gate,
    #                 qubits: Optional[List[Qubit]] = None,
    #                 additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None) -> bool:
    #     """Insert a gate into the circuit.
    #
    #     Args:
    #         circuit: The circuit onto which the gare is added.
    #         gate_to_replace: A gate instance which shall be replaced.
    #         gate_to_insert: The gate to be inserted instead.
    #         qubits: The qubits on which the gate is inserted. If None, the qubits of the
    #             reference_gate are used.
    #         additional_qubits: If qubits is None and the qubits of the reference_gate are
    #             used, this can be used to specify additional qubits before (first list in
    #             tuple) or after (second list in tuple) the qubits.
    #
    #     Returns:
    #         True, if the insertion has been successful, False otherwise.
    #     """
    #     for i, op in enumerate(circuit.data):
    #         if op[0] == gate_to_replace:
    #             circuit.data = circuit.data.pop(i) # remove gate
    #             if isinstance(gate_to_insert, IGate()):
    #                 return True
    #             #TODO check qubits placing
    #             qubits = qubits or op[1][-(gate_to_replace.num_qubits - gate_to_replace.num_clbits):]
    #             if additional_qubits:
    #                 qubits = additional_qubits[0] + qubits + additional_qubits[1]
    #             op_to_insert = (gate_to_insert, qubits, [])
    #             insertion_index = i
    #             circuit.data.insert(insertion_index, op_to_insert)
    #             return True
    #
    #     return False
