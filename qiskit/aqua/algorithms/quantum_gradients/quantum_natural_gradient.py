# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC, abstractmethod
import logging
from qiskit.aqua import AquaError, aqua_globals
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.aqua.utils.run_circuits import find_regs_by_name
import numpy as np
logger = logging.getLogger(__name__)