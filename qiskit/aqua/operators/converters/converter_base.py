# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm Base """

import logging
import numpy as np

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ConverterBase(ABC):
    """ Converters take an Operator and return a new Operator, generally isomorphic in some way with the first,
    but with certain desired properties. For example, a converter may accept a Circuit Operator and return a Sum of
    Pauli Operators representing the circuit unitary. Converters may not have polynomial space or time scaling in
    their operations. On the contrary, many converters, such as a Pauli to Matrix converter, will require
    exponential time or space unless a clever trick is known (such as the use of sparse matrices). """

    @abstractmethod
    def convert(self, operator, traverse=False):
        """ Accept the Operator and return the converted Operator """
        raise NotImplementedError
