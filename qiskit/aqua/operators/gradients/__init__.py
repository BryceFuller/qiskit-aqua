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

"""
Gradients (:mod:`qiskit.aqua.operators.gradients`)
==================================================

.. currentmodule:: qiskit.aqua.operators.gradients

This function allows gradients of operators to be computed. Gradients are often used with
optimizers, when finding a minimum, such as with :class:`~qiskit.aqua.algorithms.VQE`.

Base Classes
============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DerivativeBase
   GradientBase
   HessianBase
   QFIBase

Converters
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CircuitGradient
   CircuitQFI

Derivatives
=========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Gradient
   Hessian
   NaturalGradient
   QFI

"""

from .circuit_gradients.circuit_gradient import CircuitGradient
from .circuit_qfis.circuit_qfi import CircuitQFI
from .derivative_base import DerivativeBase
from .gradient_base import GradientBase
from .gradient import Gradient
from .natural_gradient import NaturalGradient
from .hessian_base import HessianBase
from .hessian import Hessian
from .qfi_base import QFIBase
from .qfi import QFI

__all__ = ['DerivativeBase',
           'CircuitGradient',
           'GradientBase',
           'Gradient',
           'NaturalGradient',
           'HessianBase',
           'Hessian',
           'QFIBase',
           'QFI',
           'CircuitQFI']