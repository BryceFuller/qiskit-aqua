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

from qiskit.aqua.operators.gradients.hessian.hessian import Hessian
from qiskit.aqua.operators.gradients.hessian.observable_hessian import ObservableHessian
from qiskit.aqua.operators.gradients.hessian.prob_hessian import ProbabilityHessian
from qiskit.aqua.operators.gradients.hessian.prob_hessian_lin_comb import ProbabilityHessianLinComb
from qiskit.aqua.operators.gradients.hessian.prob_hessian_param_shift import ProbabilityHessianParamShift
from qiskit.aqua.operators.gradients.hessian.state_hessian import StateHessian
from qiskit.aqua.operators.gradients.hessian.state_hessian_lin_comb import StateHessianLinComb
from qiskit.aqua.operators.gradients.hessian.state_hessian_param_shift import StateHessianParamShift

__all__ = ['Hessian',
           'ObservableHessian',
           'ProbabilityHessian',
           'ProbabilityHessianLinComb',
           'ProbabilityHessianParamShift',
           'StateHessian',
           'StateHessianLinComb',
           'StateHessianParamShift']