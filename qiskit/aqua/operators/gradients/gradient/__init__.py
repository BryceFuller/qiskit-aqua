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

from qiskit.aqua.operators.gradients.gradient.prob_gradient import Gradient
from qiskit.aqua.operators.gradients.gradient.observable_gradient import ObservableGradient
from qiskit.aqua.operators.gradients.gradient.prob_gradient_lin_comb import ProbabilityGradientAncilla
from qiskit.aqua.operators.gradients.gradient.prob_gradient_param_shift import ProbabilityGradientParamShift
from qiskit.aqua.operators.gradients.gradient.state_gradient_lin_comb import StateGradientAncilla
from qiskit.aqua.operators.gradients.gradient.state_gradient_param_shift import StateGradientParamShift

__all__ = ['Gradient',
           'ObservableGradient',
           'ProbabilityGradientAncilla',
           'ProbabilityGradientParamShift',
           'StateGradientAncilla',
           'StateGradientParamShift']
