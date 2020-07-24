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

"""The base interface for Aqua's gradients."""

from typing import Optional, Union

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import OperatorBase


class Gradient:
    """The base interface for Aqua's gradients."""

    def __init__(self, circuit: QuantumCircuit,
                 observable: Optional[OperatorBase] = None,
                 quantum_instance: Optional[Union[BaseBackend, QuantumInstance]] = None) -> None:
        """
        Args:
            circuit: The circuit for which the gradient is computed.
            observable: The observable of the expectation value underlying the gradient.
            quantum_instance: The quantum instance used to execute the circuits.
        """
        self._circuit = circuit
        self._observable = observable
        self._quantum_instance = quantum_instance

    @property
    def circuit(self) -> QuantumCircuit:
        """Return the circuit for which the gradient is computed.

        Returns:
            The circuit stored internally or None, if none is stored.
        """
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: QuantumCircuit) -> None:
        """Set the circuit for which the gradient is computed.

        Args:
            circuit: The circuit for which the gradient is computed.
        """
        self._circuit = circuit

    @property
    def observable(self) -> OperatorBase:
        """Return the observable for the expectation value w.r.t. which the gradient is computed.

        Returns:
            The observable underlying the gradient.
        """
        return self._observable

    @observable.setter
    def observable(self, observable: OperatorBase) -> None:
        """Set the observable for the expectation value w.r.t. which the gradient is computed.

        Args:
            The observable underlying the gradient.
        """
        self._observable = observable

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """Get the quantum instance of the gradient.

        Returns:
            The quantum instance stored internally or None, if none is stored.
        """

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[BaseBackend, QuantumInstance]) -> None:
        """Set the quantum instance.

        Args:
            quantum_instance: The quantum instance used to execute the circuits.
        """
        if isinstance(quantum_instance, BaseBackend):
            self._quantum_instance = QuantumInstance(quantum_instance)
        else:
            self._quantum_instance = quantum_instance

    def compute_gradient(self, parameter: Parameter) -> float:
        """Compute the gradient with respect to the provided parameter.

        Args:
            parameter: The parameter with respect to which the gradient is computed.
        """
        raise NotImplementedError
