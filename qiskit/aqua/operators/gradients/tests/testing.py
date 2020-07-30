from qiskit.aqua.operators import (StateFn, Zero, One, Plus, Minus,
                                   DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn, ListOp)
from qiskit.aqua.operators.operator_globals import H, S, I, Z
import numpy as np


from qiskit.circuit.library import RealAmplitudes, EfficientSU2
b = StateFn(RealAmplitudes(2, reps=2))
a = StateFn(EfficientSU2(2, reps=1))
list = [a, b]

op = Z ^ Z

list = ListOp(oplist=list)

exp = ~op @ list

print(exp.eval())