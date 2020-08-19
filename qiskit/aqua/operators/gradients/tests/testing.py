from qiskit.aqua.operators import (StateFn, Zero, One, Plus, Minus,
                                   DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn, ListOp, SummedOp,
                                   CircuitOp)
from qiskit.aqua.operators.operator_globals import H, S, I, Z
import numpy as np


from qiskit.circuit.library import RealAmplitudes, EfficientSU2
b = StateFn(RealAmplitudes(2, reps=2))
a = StateFn(EfficientSU2(2, reps=1))
a = a.bind_parameters(dict(zip(a.primitive.parameters, np.random.rand(len(a.primitive.parameters)).tolist())))
b = b.bind_parameters(dict(zip(b.primitive.parameters, np.random.rand(len(b.primitive.parameters)).tolist())))


op = Z ^ Z

list = [ ~op @CircuitOp(a),  ~op @CircuitOp]
exp = SummedOp(oplist=list)



# exp = ~op @ list

print(exp.eval())