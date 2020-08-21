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

"""The module for Aqua's first order derivatives."""

from .gradient import Gradient
from .operator_gradient import OperatorGradient
from .probability_gradient import ProbabilityGradient
from .probability_gradient_lin_comb import ProbabilityGradientLinComb
from .probability_gradient_param_shift import ProbabilityGradientParamShift
from .state_gradient import StateGradient
from .state_gradient_lin_comb import StateGradientLinComb
from .state_gradient_param_shift import StateGradientParamShift

__all__ = ['Gradient',
           'OperatorGradient',
           'ProbabilityGradient',
           'ProbabilityGradientLinComb',
           'ProbabilityGradientParamShift',
           'StateGradient',
           'StateGradientLinComb',
           'StateGradientParamShift']
