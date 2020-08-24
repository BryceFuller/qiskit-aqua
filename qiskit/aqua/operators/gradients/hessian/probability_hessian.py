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

"""The module implementing the probability Hessian factory."""

from qiskit.aqua.operators.gradients.hessian import Hessian


class ProbabilityHessian(Hessian):
    """The probability Hessian with respect to state gradients.

    This is a special case of the ``StateHessian`` where the hessian_operator is a projector on all
    possible basis states. This computes the hessians of the sampling probabilities of the basis
    states rather than an expectation value.
    """

    def ancilla_hessian(self, params):
        """TODO"""
        raise NotImplementedError  # TODO
