import numpy as np
from qiskit.aqua.components.optimizers import ADAM, CG, L_BFGS_B, SPSA
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit.aqua.algorithms import VQC
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils import split_dataset_to_data_and_labels
np.random.seed(42)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# training using vqc which has cross entropy loss
m = 100
# optimizer = ADAM(maxiter=m, lr=0.01)
optimizer = CG(maxiter=m, disp=True)
# optimizer = L_BFGS_B(maxiter = m)
# optimizer = SPSA(max_trials=m, c0=4.0)
backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend=backend)

num_qubits = 2
depth_varform = 2
d = num_qubits + num_qubits*depth_varform
fm = ZZFeatureMap(num_qubits, reps=2)
rx_cz = TwoLocal(depth_varform, ['rx'], 'cz', entanglement='circular')
rx_cz.draw(output='mpl')
varform = rx_cz
# varform = RealAmplitudes(num_qubits, reps=depth_varform)

from qiskit.ml.datasets import wine, ad_hoc_data

# size of training data set
training_size = 20

# size of test data set
test_size = 10
# construct training and test data
# set the following flag to True for the first data set and to False for the second dataset
use_adhoc_dataset = False
if use_adhoc_dataset:
    # first (artifical) data set to test the classifier
    _, training_input, test_input, class_labels = \
            ad_hoc_data(training_size=training_size, test_size=test_size, n=num_qubits, gap=0.3)
else:
    # second data set to test the classifier
    _, training_input, test_input, class_labels = \
            wine(training_size=training_size, test_size=test_size, n=num_qubits)

# training_input = {'A': data['A'], 'B': data['B']}
# training_data = {'A': iris.data[0:50, :], 'B':iris.data[50:100, :]}
training_data, labels = split_dataset_to_data_and_labels(training_input)
test_data, _ = split_dataset_to_data_and_labels(test_input)
vqc = VQC(optimizer, fm, varform, training_dataset=training_input, test_dataset= test_input, quantum_instance=qi,
          minibatch_size=10)
# vqc.train(data=training_data[0], labels=training_data[1], minibatch_size=10)
# result = vqc._ret
result = vqc.run()
print(result['training_loss'])
accuracy = vqc.test(test_data[0], test_data[1])
print('Accuracy', accuracy)