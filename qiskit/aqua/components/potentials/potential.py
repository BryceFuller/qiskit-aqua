# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
This module contains the definition of a base class for potentials.
"""


from abc import abstractmethod

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import Pluggable, AquaError

class Potential():

    """Base class for Potentials.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        pass
        #super().__init__()

    # @classmethod
    # def init_params(cls, params):
    #     #init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
    #     #args = {k: v for k, v in init_state_params.items() if k != 'name'}
    #     #return cls(**args)
    #     return cls(params)

    @abstractmethod
    def construct_circuit(self, mode, register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        raise NotImplementedError()

    #@property
    #def bitstr(self):
    #    return None
