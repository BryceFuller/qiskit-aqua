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

""" DerivativeBase Class """

import logging
from abc import abstractmethod
from collections.abc import Iterable as IterableAbc
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from qiskit.aqua import AquaError, QuantumInstance
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.providers import BaseBackend

from ..converters import CircuitSampler
from ..converters.converter_base import ConverterBase
from ..list_ops.composed_op import ComposedOp
from ..list_ops.list_op import ListOp
from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp
from ..state_fns.operator_state_fn import OperatorStateFn

logger = logging.getLogger(__name__)


class DerivativeBase(ConverterBase):
    r"""
    Converter for differentiating opflow objects and handling 
    things like properly differentiating combo_fn's and enforcing prodct rules
    when operator coeficients are parameterized. 

    This is distinct from CircuitGradientMethod converters which use quantum
    techniques such as parameter shifts and linear combination of unitaries
    to compute derivatives of circuits.

    CircuitGradientMethod - uses quantum techniques to get derivatives of circuits
    DerivativeBase    - uses classical techniques to differentiate opflow data strctures
    """

    # pylint: disable=arguments-differ
    @abstractmethod
    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, Parameter, List[Parameter]]] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient with respect to..

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        raise NotImplementedError

    def gradient_wrapper(self,
                         operator: OperatorBase,
                         bind_params: Union[Parameter, ParameterVector, List[Parameter]],
                         grad_params: Optional[Union[Parameter, ParameterVector, List[Parameter],
                                                     Tuple[Parameter, Parameter],
                                                     List[Tuple[Parameter, Parameter]]]] = None,
                         backend: Optional[Union[BaseBackend, QuantumInstance]] = None) \
            -> Callable[[Iterable], np.ndarray]:
        """
        Get a callable function which provides the respective gradient, Hessian or QFI for given
        parameter values. This callable can be used as gradient function for optimizers.

        Args:
            operator: The operator for which we want to get the gradient, Hessian or QFI.
            bind_params: The operator parameters to which the parameter values are assigned.
            grad_params: The parameters with respect to which we are taking the gradient, Hessian or QFI.
                        If grad_params = None, then grad_params = bind_params
            method: The method used to compute the gradient. Either 'param_shift' or 'fin_diff' or
                'lin_comb'.
            backend: The quantum backend or QuantumInstance to use to evaluate the gradient,
                Hessian or QFI.

        Returns:
            callable(param_values): Function to compute a gradient, Hessian or QFI. The function
            takes an Iterable as argument which holds the parameter values.
        """
        if not grad_params:
            grad_params = bind_params
            # raise AquaError(
            #     'Please define parameters for which the gradient/Hessian/QFI shall be computed')

        def gradient_fn(p_values):
            p_values_dict = dict(zip(bind_params, p_values))
            if not backend:
                converter = self.convert(operator, grad_params).assign_parameters(p_values_dict)
            else:
                if isinstance(backend, QuantumInstance):
                    if backend.is_statevector:
                        converter = self.convert(operator, grad_params).assign_parameters(p_values_dict)
                    else:
                        p_values_dict = {k: [v] for k, v in p_values_dict.items()}
                        converter = CircuitSampler(backend=backend).convert(
                            self.convert(operator, grad_params), p_values_dict)
                else:
                    if backend.name().startswith('statevector'):
                        converter = self.convert(operator, grad_params).assign_parameters(p_values_dict)
                    else:
                        p_values_dict = {k: [v] for k, v in p_values_dict.items()}
                        converter = CircuitSampler(backend=backend).convert(
                            self.convert(operator, grad_params), p_values_dict)
                return converter.eval()

        return gradient_fn

    @staticmethod
    def parameter_expression_grad(param_expr: ParameterExpression,
                                  param: Parameter) -> ParameterExpression:

        """Get the derivative of a parameter expression w.r.t. the given parameter.

        Args:
            param_expr: The Parameter Expression for which we compute the derivative
            param: Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
        """
        if not isinstance(param_expr, ParameterExpression):
            # TODO: return ParameterExpression
            return 0.0

        if param not in param_expr._parameter_symbols:
            # TODO: return ParameterExpression
            return 0.0

        import sympy as sy
        expr = param_expr._symbol_expr
        keys = param_expr._parameter_symbols[param]
        expr_grad = sy.N(0)
        if not isinstance(keys, IterableAbc):
            keys = [keys]
        for key in keys:
            expr_grad += sy.Derivative(expr, key).doit()
        return ParameterExpression(param_expr._parameter_symbols, expr=expr_grad)

    @classmethod
    def _erase_operator_coeffs(cls, operator: OperatorBase, coeff: float = 1.0) -> OperatorBase:
        """TODO

        Args:
            operator: TODO
            coeff: TODO

        Returns:
            TODO

        """
        if isinstance(operator, PrimitiveOp):
            return operator / operator._coeff
        elif isinstance(operator, OperatorBase):
            op_coeff = operator._coeff
            return (operator / op_coeff).traverse(
                partial(cls._erase_operator_coeffs, coeff=coeff * op_coeff))
        return operator * coeff

    @classmethod
    def _factor_coeffs_out_of_composed_op(cls, operator: OperatorBase) -> OperatorBase:
        """TODO

        Args:
            operator: TODO

        Returns:
            TODO

        Raises:
            ValueError: TODO
        """
        if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
            return operator.traverse(cls._factor_coeffs_out_of_composed_op)
        if isinstance(operator, ComposedOp):
            total_coeff = 1.0
            take_norm_of_coeffs = False
            for op in operator.oplist:

                if take_norm_of_coeffs:
                    total_coeff *= (op._coeff * np.conj(op._coeff))
                else:
                    total_coeff *= op._coeff
                if hasattr(op, 'primitive'):
                    prim = op.primitive
                    if isinstance(prim, ListOp):
                        raise ValueError("This operator was not properly decomposed. "
                                         "By this point, all operator measurements should "
                                         "contain single operators, otherwise the coefficient "
                                         "gradients will not be handled properly.")
                    if hasattr(prim, 'coeff'):
                        if take_norm_of_coeffs:
                            total_coeff *= (prim._coeff * np.conj(prim._coeff))
                        else:
                            total_coeff *= prim._coeff

                if isinstance(op, OperatorStateFn) and op._is_measurement:
                    take_norm_of_coeffs = True
            return total_coeff * cls._erase_operator_coeffs(operator)

        else:
            return operator
