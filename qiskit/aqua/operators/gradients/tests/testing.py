from qiskit.aqua.operators import (StateFn, Zero, One, Plus, Minus,
                                   DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn, ListOp, SummedOp,
                                   CircuitOp)
from qiskit.aqua.operators.operator_globals import H, S, I, Z
import numpy as np


from qiskit.circuit.library import RealAmplitudes, EfficientSU2


def unroll_operator(operator):
    def unroll_traverse(operator):
        if isinstance(operator, ListOp):
            # Traverse the elements in the ListOp
            res = [op.traverse(unroll_traverse) for op in operator]
            # Separate out the lists from non-list elements
            lists = [l for l in res if isinstance(l, (list, ListOp))]
            not_lists = [r for r in res if not isinstance(r, (list, ListOp))]
            # unroll the list elements and recombine everything
            unrolled = [y for x in lists for y in x]
            res = not_lists + unrolled
            return res
        return operator

    # When unroll_traverse terminates, there will still be
    # one last layer of nested lists to unroll. (computational tree will be depth <=2)
    unrolled_op = operator.traverse(unroll_traverse)
    lists = [l for l in unrolled_op if isinstance(l, (list, ListOp))]
    not_lists = [r for r in unrolled_op if not isinstance(r, (list, ListOp))]
    # unroll the list elements and recombine everything
    unrolled = [y for x in lists for y in x]
    return not_lists + unrolled

b = StateFn(RealAmplitudes(2, reps=2))
a = StateFn(EfficientSU2(2, reps=1))
a = a.bind_parameters(dict(zip(a.primitive.parameters, np.random.rand(len(a.primitive.parameters)).tolist())))
b = b.bind_parameters(dict(zip(b.primitive.parameters, np.random.rand(len(b.primitive.parameters)).tolist())))


op = Z ^ Z

list = [a, b]
exp = ListOp(oplist=list)

exp = ~Zero @ list
c = exp.traverse(unroll_operator)

print(exp.eval())