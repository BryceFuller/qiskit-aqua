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

"""The module for Aqua's gradient."""

from .derivatives_base import DerivativeBase
from .circuit_gradient_methods.circuit_gradient_method import CircuitGradientMethod

from .qfi import QFI
from .natural_gradient import NaturalGradient
from .gradient import Gradient
from .hessian import Hessian

from .circuit_gradient_methods import (LinCombGradient, ParamShiftGradient,
                                       LinCombQFI, BlockDiagQFI, DiagQFI)

__all__ = ['DerivativeBase',
           'CircuitGradientMethod',
           'Gradient',
           'NaturalGradient',
           'Hessian',
           'QFI',
            'LinCombGradient',
           'ParamShiftGradient',
           'LinCombQFI',
           'BlockDiagQFI',
           'DiagQFI'
           ]