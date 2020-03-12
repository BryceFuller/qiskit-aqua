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
import networkx as nx
import itertools

from .evolution_base import EvolutionBase

from qiskit.aqua.operators import (OpVec, OpSum, OpPauli, OpPrimitive, Z, I, PauliChangeOfBasis, AbelianGrouper)

from . import OpEvolution
from .trotterizations import TrotterizationBase

logger = logging.getLogger(__name__)


class PauliTrotterEvolution(EvolutionBase):
    """ TODO

    """

    def __init__(self, trotter_mode='suzuki', reps=1, group_paulis=True):
        """
        Args:

        """

        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            self._trotter = TrotterizationBase.factory(mode=trotter_mode, reps=reps)

        self._grouper = AbelianGrouper() if group_paulis else None

    @property
    def trotter(self):
        return self._trotter

    @trotter.setter
    def trotter(self, trotter):
        self._trotter = trotter

    def convert(self, operator):
        if self._grouper:
            # Sort into commuting groups
            operator = self._grouper.convert(operator).reduce()
        return self._recursive_convert(operator)

    def _recursive_convert(self, operator):
        if isinstance(operator, OpEvolution):
            if isinstance(operator.primitive, OpSum):
                # if operator.primitive.abelian:
                #     return self.evolution_for_abelian_paulisum(operator.primitive)
                # else:
                trotterized = self.trotter.trotterize(operator.primitive)
                return self._recursive_convert(trotterized)
            elif isinstance(operator.primitive, OpPauli):
                return self.evolution_for_pauli(operator.primitive)
            # Covers OpVec, OpComposition, OpKron
            elif isinstance(operator.primitive, OpVec):
                converted_ops = [self._recursive_convert(op) for op in operator.primitive.oplist]
                return operator.__class__(converted_ops, coeff=operator.coeff)
        elif isinstance(operator, OpVec):
            return operator.traverse(self.convert).reduce()
        else:
            return operator

    def evolution_for_pauli(self, pauli_op):
        # TODO Evolve for group of commuting paulis, TODO pauli grouper

        def replacement_fn(cob_instr_op, dest_pauli_op):
            z_evolution = dest_pauli_op.exp_i()
            # Remember, circuit composition order is mirrored operator composition order.
            return cob_instr_op.adjoint().compose(z_evolution).compose(cob_instr_op)

        # Note: PauliChangeOfBasis will pad destination with identities to produce correct CoB circuit
        destination = Z * pauli_op.coeff
        cob = PauliChangeOfBasis(destination_basis=destination, replacement_fn=replacement_fn)
        return cob.convert(pauli_op)

    # TODO
    @staticmethod
    def compute_cnot_distance(pauli_op1, pauli_op2):
        sig_pauli1_bits = np.logical_and(pauli_op1.primitive.z, pauli_op1.primitive.x)
        sig_pauli2_bits = np.logical_and(pauli_op2.primitive.z, pauli_op2.primitive.x)

        # Has anchor case
        if any(np.logical_and(sig_pauli1_bits, sig_pauli2_bits)):
            # All the equal bits cost no cnots
            non_equal_sig_bits = np.logical_xor(sig_pauli1_bits, sig_pauli2_bits)
            # Times two because we need cnots to anchor and back
            return 2 * np.sum(non_equal_sig_bits)
        # No anchor case
        else:
            # Basically just taking each to and from the identity
            cnot_cost_p1 = np.abs(np.sum(sig_pauli1_bits) - 1)
            cnot_cost_p2 = np.abs(np.sum(sig_pauli2_bits) - 1)
            return 2 * (cnot_cost_p1 + cnot_cost_p2)

    # TODO
    def evolution_for_abelian_paulisum(self, op_sum):
        if not all([isinstance(op, OpPauli) for op in op_sum.oplist]):
            raise TypeError('Evolving abelian sum requires Pauli elements.')

        cob = PauliChangeOfBasis()

        pauli_graph = nx.Graph()
        pauli_graph.add_nodes_from(op_sum.oplist)
        pauli_graph.add_weighted_edges_from([(ops[0], ops[1], self.compute_cnot_distance(ops[0], ops[1]))
                                             for ops in itertools.combinations(op_sum.oplist, 2)])
        print(pauli_graph.edges(data=True))
        tree = nx.minimum_spanning_tree(pauli_graph)
        tree_edges = nx.dfs_edges(tree)
