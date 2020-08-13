from qiskit.aqua.operators.gradients.gradient.prob_gradient_lin_comb import ProbabilityGradientLinComb
from qiskit.aqua.operators.gradients.gradient.state_gradient_lin_comb import StateGradientLinComb

from qiskit import BasicAer
from qiskit.aqua.operators import X, Z, StateFn, CircuitStateFn, CircuitSampler
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import numpy as np

from qiskit.aqua import QuantumInstance, aqua_globals


aqua_globals.random_seed = 50
# Set quantum instance to run the quantum generator
qi = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                          seed_simulator=2,
                          seed_transpiler=2)

H = 0.5 * X - 1 * Z
a = Parameter('a')
b = Parameter('b')
params = [a, b]

q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.h(q)
qc.rz(params[0], q[0])
qc.rx(params[1], q[0])

op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

state_grad = StateGradientLinComb().convert(operator=op, params=params)
values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
               {params[0]: np.pi / 2, params[1]: np.pi / 4}]
correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                  [-0.5, -1 / np.sqrt(2)]]
for i, value_dict in enumerate(values_dict):
    print(state_grad.assign_parameters(value_dict).eval())
# converter = CircuitSampler(backend=qi).convert(state_grad)
a = 0



