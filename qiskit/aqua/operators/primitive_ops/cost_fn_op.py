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

""" CostFnOp Class """

from typing import Union, Set, Dict, Optional, cast, Callable
import logging
import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import RZGate, RYGate, RXGate, XGate, YGate, ZGate, IGate

from ..operator_base import OperatorBase
from .primitive_op import PrimitiveOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..legacy.weighted_pauli_operator import WeightedPauliOperator

logger = logging.getLogger(__name__)
PAULI_GATE_MAPPING = {'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'I': IGate()}


class CostFnOp(PrimitiveOp):
    """ Class for Operators backed by Terra's ``Pauli`` module.

    """

    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 cost_fn: Callable = None,
                 grad_cost_fn: Callable = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
            Args:
                primitive: The Pauli which defines the behavior of the underlying function.
                coeff: A coefficient multiplying the primitive.

            Raises:
                TypeError: invalid parameters.
        """
        self.cost_fn = cost_fn
        self.grad_cost_fn = grad_cost_fn
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {'Pauli'}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, PauliOp) and self.primitive == other.primitive:
            return PauliOp(self.primitive, coeff=self.coeff + other.coeff)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return ValueError("Adjoint of a cost function not defined")

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, PauliOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:

        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              cost_fn=self.cost_fn,
                              grad_cost_fn=self.grad_cost_fn,
                              coeff=self.coeff * scalar)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # Both Paulis
        if isinstance(other, PauliOp):
            # Copying here because Terra's Pauli kron is in-place.
            op_copy = Pauli(x=other.primitive.x, z=other.primitive.z)  # type: ignore
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            return PauliOp(op_copy.kron(self.primitive), coeff=self.coeff * other.coeff)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        if isinstance(other, CircuitOp):
            return self.to_circuit_op().tensor(other)

        return TensoredOp([self, other])
    """
    def compose(self, other: OperatorBase) -> OperatorBase:
        other = self._check_zero_for_composition_and_expand(other)

        # Both Paulis
        if isinstance(other, PauliOp):
            product, phase = Pauli.sgn_prod(self.primitive, other.primitive)
            return PrimitiveOp(product, coeff=self.coeff * other.coeff * phase)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        if isinstance(other, (CircuitOp, CircuitStateFn)):
            return self.to_circuit_op().compose(other)

        return super().compose(other)
    """
    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff  # type: ignore

    def to_spmatrix(self) -> spmatrix:
        """ Returns SciPy sparse matrix representation of the Operator.

        Returns:
            CSR sparse matrix representation of the Operator.

        Raises:
            ValueError: invalid parameters.
        """
        return self.primitive.to_spmatrix() * self.coeff  # type: ignore

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self,
             front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase]] = None
             ) -> Union[OperatorBase, float, complex]:
        if front is None:
            return self.to_matrix_op()

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ..state_fns.state_fn import StateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..list_ops.list_op import ListOp
        from .circuit_op import CircuitOp

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                        for front_elem in front.oplist])


        # Need an ListOp-specific carve-out here to make sure measurement over a ListOp doesn't
        # produce two-dimensional ListOp from composing from both sides of primitive.
        # Can't use isinstance because this would include subclasses.
        # pylint: disable=unidiomatic-typecheck
        if type(front) == ListOp:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])  # type: ignore

        if isinstance(front,  DictStateFn):
            in_place_eval = {key: self.primitive.eval(key).adjoint().eval(key) \
                                  * front.primitive[key]                       \
                                  * self.coeff for key in front.primitive} # type: ignore
            return DictStateFn(in_place_eval, coeff=front.coeff)

        return new_front

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..list_ops.list_op import ListOp
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return self.traverse(lambda x: x.assign_parameters(param_dict), coeff=param_value)

    def exp_i(self) -> OperatorBase:
        """ Return a ``CircuitOp`` equivalent to e^-iH for this operator H. """
        # if only one qubit is significant, we can perform the evolution
        corrected_x = self.primitive.x[::-1]  # type: ignore
        corrected_z = self.primitive.z[::-1]  # type: ignore
        # pylint: disable=import-outside-toplevel,no-member
        sig_qubits = np.logical_or(corrected_x, corrected_z)
        if np.sum(sig_qubits) == 0:
            # e^I is just a global phase, but we can keep track of it! Should we?
            # For now, just return identity
            return PauliOp(self.primitive)
        if np.sum(sig_qubits) == 1:
            sig_qubit_index = sig_qubits.tolist().index(True)
            coeff = np.real(self.coeff) \
                if not isinstance(self.coeff, ParameterExpression) \
                else self.coeff
            # Y rotation
            if corrected_x[sig_qubit_index] and corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RYGate(coeff))
            # Z rotation
            elif corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RZGate(coeff))
            # X rotation
            elif corrected_x[sig_qubit_index]:
                rot_op = PrimitiveOp(RXGate(coeff))

            from ..operator_globals import I
            left_pad = I.tensorpower(sig_qubit_index)
            right_pad = I.tensorpower(self.num_qubits - sig_qubit_index - 1)
            # Need to use overloaded operators here in case left_pad == I^0
            return left_pad ^ rot_op ^ right_pad
        else:
            from ..evolutions.evolved_op import EvolvedOp
            return EvolvedOp(self)

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = None
                 ) -> OperatorBase:
        r"""
        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in
        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.

        Args:
            convert_fn: The function to apply to the internal OperatorBase.
            coeff: A coefficient to multiply by after applying convert_fn.

        Returns:
            The converted StateFn.
        """
        if isinstance(self.primitive, OperatorBase):
            return CostFnOp(convert_fn(self.primitive),
                            cost_fn=self.cost_fn,
                            grad_cost_fn=self.grad_cost_fn,
                            coeff=coeff or self.coeff)
        else:
            return self