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

""" CostFnMeasurement Class """


from typing import Union, Set, Dict, Optional, cast, Callable
import numpy as np

from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from .state_fn import StateFn
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp


# pylint: disable=invalid-name

class CostFnMeasurement(StateFn):
    r"""
    A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 cost_fn: Callable = None,
                 grad_cost_fn: Callable = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = True) -> None:
        """
        Args:
            primitive: The ``OperatorBase`` which defines the behavior of the underlying State
                function.
            coeff: A coefficient by which to multiply the state function
            is_measurement: Whether the StateFn is a measurement operator
        """
        if primitive is None and cost_fn is None:
            raise ValueError
        if not is_measurement:
            raise ValueError("CostFnMeasurement is only defined as a measurement")

        self.cost_fn = cost_fn
        self.grad_cost_fn = grad_cost_fn
        super().__init__(primitive, coeff=coeff, is_measurement=True)

    def primitive_strings(self) -> Set[str]:
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        if hasattr(self.primitive, 'num_qubits'):
            return self.primitive.num_qubits
        else:
            return None

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        return ValueError("Adjoint of a cost function not defined")

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:

        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              cost_fn=self.cost_fn,
                              grad_cost_fn=self.grad_cost_fn,
                              coeff=self.coeff * scalar,
                              is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        return OperatorStateFn(self.primitive.to_matrix_op(massive=massive) * self.coeff,
                               is_measurement=self.is_measurement)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        r"""
        Not Implemented
        """
        raise NotImplementedError


    def to_circuit_op(self) -> OperatorBase:
        r""" Return ``StateFnCircuit`` corresponding to this StateFn. Ignore for now because this is
        undefined. TODO maybe call to_pauli_op and diagonalize here, but that could be very
        inefficient, e.g. splitting one Stabilizer measurement into hundreds of 1 qubit Paulis."""
        raise NotImplementedError

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('CostFnMeasurement', prim_str)
        else:
            return "{}({}) * {}".format(
                'CostFnMeasurement',
                prim_str,
                self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        #Convert front into a dict
        #Save the type 
        from .dict_state_fn import DictStateFn
        # Need an ListOp-specific carve-out here to make sure measurement over a ListOp doesn't
        # produce two-dimensional ListOp from composing from both sides of primitive.
        # Can't use isinstance because this would include subclasses.
        # pylint: disable=unidiomatic-typecheck
        if type(front) == ListOp:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])  # type: ignore
        if self.primitive is not None:
            return front.adjoint().eval(self.primitive.eval(front)) * self.coeff  # type: ignore
        elif isinstance(front, DictStateFn):
             data = front.primitive
             return self.cost_fn(data)
            

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        raise NotImplementedError
