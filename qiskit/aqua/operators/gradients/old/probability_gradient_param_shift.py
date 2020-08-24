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

"""The module to compute the probability gradients with the parameter shift method."""

from typing import Optional, Union, List, Dict
from functools import partial

from qiskit import Aer
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.circuit import Parameter, ParameterVector

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.converters.circuit_sampler import CircuitSampler
from qiskit.aqua.operators.gradients.gradient.gradient_param_shift import GradientParamShift


class ProbabilityGradientParamShift(GradientParamShift):
    """Special Case of the StateGradient where the gradient_operator is the identity.

    Actually there are some more nuanced differences on how this is computed in practice!
    """

    def convert(self,
                operator: OperatorBase,
                params: Union[Parameter, ParameterVector, List[Parameter]] = None,
                param_bindings: Optional[Dict[Parameter, float]] = None,
                backend: Optional[Union[QuantumInstance, BaseBackend]] = None,
                analytic: bool = True) -> OperatorBase:
        r"""
        Args
            operator: The operator corresponding to our quantum state we are taking the
                gradient of: prepares |ψ(ω)〉.
            params: The parameters we are taking the gradient with respect to
            param_bindings: TODO
            backend: The backend for the circuit sampler. If None, Aer's QASM backend is chosen
                by default with 1024 shots.
            analytic: If True compute an analytic gradient, else compute a finite difference
                approximations.
        """
        # TODO add finite differences
        if analytic is False:
            raise NotImplementedError('Finite difference scheme not implemented yet.')

        # TODO backend and param bindings?
        if operator.is_measurement:
            raise AquaError('Probability gradients are computed with respect to states instead of '
                            'expectation values. Please remove the measurement operator.')

        operator = super().convert(operator, params)

        if backend is None:
            # TODO don't rely on Aer, user might not have installed this, rather just use eval
            # and not the circuit sampler
            backend = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                      shots=1024)

        # TODO: CHECK if state_operator contains an operator measurement, throw an error if it does!
        sampler = CircuitSampler(backend=backend)

        # The CircuitStateFn's need to be wrapped so that the coefficient
        # is stored elsewhere during circuit sampling
        wrapped_op = self._wrap_circuit_statefn(operator)
        sampled_op = sampler.convert(wrapped_op.bind_parameters(param_bindings))
        grad_op = self._ravel(sampled_op).reduce()

        if isinstance(grad_op, ListOp):
            gradients = [grad.primitive for grad in grad_op.reduce()]
        else:
            gradients = [grad_op.primitive]
        return gradients

    def _wrap_circuit_statefn(self, operator):
        if isinstance(operator, StateFn):
            return ListOp([operator / operator.coeff], coeff=operator.coeff)
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self._wrap_circuit_statefn))
        return operator

    def _ravel(self, operator):
        if isinstance(operator, ListOp):
            if len(operator.oplist) == 1:
                return operator.oplist[0] * operator.coeff
            else:
                return operator.traverse(partial(self._ravel))
        return operator
