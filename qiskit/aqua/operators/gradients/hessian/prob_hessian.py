
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

""" Hessian Class """

from typing import Optional, Callable, Union, List, Tuple
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.aqua.operators.gradients.hessian import Hessian

from qiskit.aqua.operators.operator_base import OperatorBase
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.primitive_ops.pauli_op import PauliOp
from qiskit.aqua.operators.primitive_ops.circuit_op import CircuitOp
from qiskit.aqua.operators.list_ops.list_op import ListOp
from qiskit.aqua.operators.list_ops.composed_op import ComposedOp
from qiskit.aqua.operators.state_fns.state_fn import StateFn
from qiskit.aqua.operators.operator_globals import H, S, I
from ..gradient_base import GradientBase
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector

logger = logging.getLogger(__name__)

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

""" ProbabilityHessian Class """


class ProbabilityHessian(Hessian):
    r"""
    Special Case of the StateHessian where the hessian_operator is a projector on all possible basis states.
    This computes the hessians of the sampling probabilities of the basis states rather than an expectation value.

    We are interested in computing:
    d⟨ψ(ω)|ψ(ω)〉/ dω for ω in params
    """
