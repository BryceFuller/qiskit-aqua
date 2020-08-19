# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

### Variational Quantum Time Evolution (VarQTE)
"""
This algorithm applies variational quantum time evolution to a given input state.
Given a parametrized quantum circuit, the Ansatz, the parameters are updated by using McLachlan`s variational principle
such that the parameter evolution mimics the time evolution of a given Hamiltonian.

- S. McArdle, T. Jones, S. Endo, Y. Li, S. C. Benjamin, and X. Yuan,
  “Variational ansatz-based quantum simulation of imaginary time evolution,” npj Quantum Information,
  vol. 5, no. 1, p. 75, 2019.
- X. Yuan, S. Endo, Q. Zhao, S. Benjamin, and Y. Li, “Theory of variational quantum simulation,” Quantum, vol. 3, 191,
  2019.

"""
from typing import Optional
import csv
import os
import logging

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua.algorithms import QuantumAlgorithm

logger = logging.getLogger(__name__)


class VarQTE(QuantumAlgorithm):
    """
    Variational Simulation of Imaginary Time Evolution
    arXiv:1804.03023

    """

    def __init__(self, regularization=None, imaginary_hamiltonian=None, real_hamiltonian=None, tau=1,
                 num_time_steps=500, ansatz=None, initial_state=None, omega_0=None, q_target=None, ODE_Solver=None,
                 gibbs_prep=False, global_phase_fix=False):
        """
        Initialize VarQTE Algorithm
        Args:
            regularization(None or str: {'ridge', 'lasso', 'perturb_diag'}):
                if None no regularization applied in _get_dH_dt_weightsdt_weights but small perturbations are
                applied on the left side of the SLE and the solution is computed with lstsq,
                else use ridge or lasso with automatic optimal parameter search
                measurement_error_mitigation (Bool): Use measurement error mitigation (deprecated for statevector
                simulator) - for further information see
                `Qiskit Tutorials <https://github.com/Qiskit/qiskit-iqx-tutorials/>`_
            imaginary_hamiltonian (WeightedPauliOperator): Hamiltonian for imaginary time evolution
            real_hamiltonian (WeightedPauliOperator): Hamiltonian for real time evolution
            tau (float): time for imaginary time evolution
            num_time_steps(int): number of time_steps
            ansatz (Ansatz): parametrized variational circuit (Ansatz)
                    Notably, the following operator construction is correct iff the parametrized gates in the Ansatz
                    circuit (construct_circuit) are Pauli-rotations or controlled Pauli rotations.
            initial_state (QuantumCircuit/Instruction): Initial state, if None set to $|0>^{\otimes \text{num_qubits}}$
            omega_0 (array): Initial parameters for the Ansatz circuit
            q_target (QuantumRegister): Register to attach the Ansatz too. If None create a Quantum Register with
                                        number qubits = hamiltonian.num_qubits
            ODE_Solver (callable): ODE Solver to be used to propagate the Ansatz weights
            gibbs_prep (Boolean): True - VarQTE is used for QRBM state preparation, False - else
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                        for a potential mismatch between the target and the trained state - see
                        `Theory of variational quantum simulation
                         <https://quantum-journal.org/papers/q-2019-10-07-191/>`_
        """


    def _get_dt_weights(self, A, C):
        """
        Args:
            A: operator described in arXiv:1804.03023
            C: operator described in arXiv:1804.03023

        Returns: derivative of the Ansatz circuit's weights (parameters) w.r.t. the time step dt

        """

    def dt_weights(omega_t):
        """
	Wrapper, needed for ODE solvers

        Args:
	    omega_t: current parameters

        Returns: derivative of the Ansatz circuit's weights (parameters) w.r.t. to time

        """
        A = get_A(omega_t)
        C = get_C(omega_t)

        return self._get_dt_weights(A, C)

    def _get_dH_dt_weights(self, A, dH_A, dH_C, dt_weights):  # only needed for QBM with Gibbs State preparation
        """
        Args:
            A: operator described in arXiv:1804.03023
            dH_A: derivative of A w.r.t. the Hamiltonian weights (parameters)
            dH_C: derivative of C w.r.t. the Hamiltonian weights (parameters)

        Returns: derivative of the Ansatz circuit's weights (parameters) time derivative w.r.t. the Hamiltonian weights
        (parameters)

        """

    def _propagate_weights(self, dt_weights):
        """
        Propagate the weights for time step dt

        Args:
            dt_weights:  derivative of the Ansatz circuit's weights (parameters) w.r.t. the time step dt

        Returns:

        """

    def _propagate_dH_weights(self, dH_dt_weights):
        """
        Calculate the derivative of the Ansatz circuit's weights (parameters) w.r.t. the Hamiltonian weights
        (parameters)

        Args:
            dt_weights:  derivative of the Ansatz circuit's weights (parameters) w.r.t. the time step dt

        Returns:

        """

    def _store_params(self, ansatz, depth, num_step, total_steps, fidelity, params, dH_params):
        """
        Stores parameters in a csv file in self._snapshot_dir
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                    propagation
            depth (int): depth of Ansatz circuit
            num_step (int):  currently conducted time steps
            total_steps (int): total number of time steps to be conducted
            fidelity (float): fidelity from trained to target state
            params (array-like): current parameters for the Ansatz
            dH_params (array-like):current derivative of the parameters for the Ansatz w.r.t. the Hamiltonian parameters

        Returns:

        """

    def run(self, snapshot_dir=None, resume=False):
        r"""
        Run Var SITE Algorithm
        Args:
            snapshot_dir (Union(str, None)): path or None, if path given store cvs file
                                      with parameters to the directory
            resume(Bool): if True and snapshot_dir given, load params from snapshot_dir and resume algorithm
        """

    def _run(self):
        """
        Algorithm is executed and the parameters are updated

        """


class VarSITEAnsatz(VariationalForm, ABC):

    """
    Base class for VarSITE Operators
    Notably, the following operator construction is correct iff the parametrized gates in the Ansatz circuit
    (construct_circuit) are Pauli-rotations or controlled Pauli rotations.
    """
    @abstractmethod
    def __init__(self, initial_state, num_qubits, depth, q_target=None, hamiltonian=None, global_phase_fix=False,
                 quantum_instance=None):
        """

        Args:
            initial_state (QuantumCircuit): Initial state given as QuantumCircuit with a single QuantumRegister
            num_qubits (int): number of qubits used in the Ansatz circuit
            depth (int): depth Ansatz circuit
            q_target (QuantumRegister): qubits to attach the Ansatz too. If None use the first Quantum Register from
            the initial_state.
            hamiltonian (WeightedPauliOperator): Hamiltonian for imaginary time evolution
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                                    for a potential mismatch between the target and the trained state - see
                                    Note that the additional parameter is the LAST parameter in the array
                                    `Theory of variational quantum simulation
                                     <https://quantum-journal.org/papers/q-2019-10-07-191/>`_
            quantum_instance(QuantumInstance): Quantum Instance used for running the circuits

        """
        super().__init__()
        self._quantum_instance = quantum_instance
        self._hamiltonian = hamiltonian
        if initial_state:
            self._initial_state = initial_state
        else:
            q = QuantumRegister(num_qubits)
            self._initial_state = QuantumCircuit(q)
        self._num_qubits = num_qubits
        if q_target:
            self._q_target = q_target
        else:
            self._q_target = self._initial_state.qregs[0]
        self._depth = depth
        self._C_circs = []
        self._global_phase_fix = global_phase_fix

    @property
    def hamiltonian(self):
        """
        Get Hamiltonian
        Returns:Hamiltonian for which the imaginary time evolution shall be simulated with
            the VarForm

        """
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, h):
        """
        Set Hamiltonian
        Args:
            h (WeightedPauliOperator): Hamiltonian for which the imaginary time evolution shall be simulated with
            the VarForm

        Returns:

        """
        self._hamiltonian = h
        return

    @property
    def num_qubits(self):
        """Number of qubits of the variational form.

        Returns:
            An integer indicating the number of qubits.
        """
        return self._num_qubits

    @property
    def q_target(self):
        """List of target qubit registers on which the variational form is supposed to act.

        Returns:
            An integer indicating the number of qubits.
        """
        return self._q_target

    @property
    def depth(self):
        """Depth of the variational form.

        Returns:
            An integer indicating the depth.
        """
        return self._depth

    @depth.setter
    def depth(self, d):
        """
        Set Ansatz Depth
        Args:
            d: Int, depth

        Returns:

        """
        self._depth = d
        return

    @property
    def global_phase_fix(self):
        """global phase fix parameter

        Returns:
            A Boolean indicating whether an ancilla is added to circumvent potential mismatch in the global phase
            of the trained and target Gibbs state.
        """
        return self._global_phase_fix

    @global_phase_fix.setter
    def global_phase_fix(self, phase_fix):
        """
        Set global phase fix parameter
        Args:
            phase_fix: Bool

        Returns:

        """
        self._global_phase_fix = phase_fix
        return

    @property
    def quantum_instance(self):
        """ returns quantum instance """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, qi):
        """
        Args:
            qi: QuantumInstance

        Returns:
        """
        self._quantum_instance = qi

    def get_exp_value(self, circuit):
        r"""
        Evaluate the expectation value $\langle Z \rangle$ w.r.t. the ancilla qubit (named 'q')
        Args:
            circuit(QuantumCircuit): with a single qubit QuantumRegister with name 'q'

        Returns: expectation value $\langle Z \rangle$ w.r.t. the 'q' QuantumRegister

        """

        def prepare(qc, single_qubit_mem=False):
            r"""
            Get the operator and the corresponding QuantumRegister to evaluate $\langle Z \rangle$
            Args:
                qc (QuantumCircuit): This circuit prepares the state for which we want to evaluate $\langle Z \rangle$
                for the QuantumRegister named 'q'
            Returns:
                WeightedPauliOperatorqubit_op

            """
            evaluation_qubit = find_regs_by_name(qc, 'q')
            if single_qubit_mem:
                pauli = 'Z'
                qregs_list = evaluation_qubit
            else:
                qregs_list = circuit_item.qregs
                index_evaluation_qubit = qregs_list.index(evaluation_qubit)
                pauli = ''
                for i, qreg in enumerate(qregs_list):
                    if i == index_evaluation_qubit:
                        pauli = 'Z' + pauli
                    else:
                        pauli = 'I' * qreg.size + pauli

                temp = []
                for element in qregs_list:
                    for i in range(element.size):
                        temp.append(element[i])
                qregs_list = temp

            qubit_op = WeightedPauliOperator([[1, Pauli.from_label(pauli)]])
            return qubit_op, qregs_list
        sv_mode = False
        if self._quantum_instance.is_statevector:
            sv_mode = True
        if isinstance(circuit, list):
            qc = []
            qubit_ops = []
            for k, circuit_item in enumerate(circuit):
                if self._quantum_instance.measurement_error_mitigation_cls is not None:
                    qubit_op, qregs_list = prepare(circuit_item, single_qubit_mem=False)
                else:
                    qubit_op, qregs_list = prepare(circuit_item, single_qubit_mem=True)
                qc.extend(qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode, wave_function=circuit_item,
                                                                qr=qregs_list, circuit_name_prefix='circuits'+str(k)))
                qubit_ops.append(qubit_op)
            # try:
            #     initial_layout = self._quantum_instance.compile_config['initial_layout']
            #     temp = {}
            #     for k, layout_item in enumerate(initial_layout):
            #         temp[qregs_list[k]] = layout_item
            #     self._quantum_instance.set_config(initial_layout = temp)
            # except Exception:
            #     pass
        else:
            qubit_op, qregs_list = prepare(circuit)
            qc = qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode, wave_function=circuit, qr=qregs_list)
                                                  # qr=find_regs_by_name(circuit, 'q'))

        success = False
        counter = 0
        while not success:
            try:
                result = self._quantum_instance.execute(qc)
                success = True
            except Exception:
                counter += 1
            if counter > 10:
                raise AquaError('Get expectation value failed.')
                break
        if isinstance(circuit, list):
            avg = []
            for k, circuit_item in enumerate(circuit):
                avg_temp, _ = qubit_ops[k].evaluate_with_result(statevector_mode=sv_mode, result=result,
                                                               circuit_name_prefix='circuits'+str(k))
                avg.append(avg_temp)
        else:
            avg, _ = qubit_op.evaluate_with_result(statevector_mode=sv_mode, result=result)

        return avg

    @abstractmethod
    def construct_circuit(self, parameters, q=None):
        """
        Construct Ansatz circuit applied to initital state
        Args:
            parameters (numpy.ndarray[float]): circuit parameters.
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            Ansatz circuit applied to initital state (QC)

        """
        raise NotImplementedError

    @abstractmethod
    def _get_A_entry_circuit(self, i, j, omega_t):
        r"""
        Get QuantumCircuit to compute the matrix element $A_{i,j}$ for the current circuit parameters $\omega_t$
        Args:
            i (int): index to denote the ith row of A
            j (int): index to denote the jth column of A
            omega_t (array-like): current circuit parameters

        Returns:
            Quantum Circuit

        """
        raise NotImplementedError

    def get_A(self, omega_t):
        r"""
        Get the matrix $A$ for the current circuit parameters $\omega_t$
        Args:
            omega_t (array-like): current circuit parameters

        Returns:
            array: $A$

        """
        fifj = 1 / 4.
        A = np.zeros((len(omega_t), len(omega_t)))

        A_circuits = []
        for i in range(len(omega_t)):
            j = 0
            while j <= i:
                if j==i:
                    A[i, j] = fifj
                else:
                    A_circuits.append(self._get_A_entry_circuit(i, j, omega_t))
                j+=1
        A_exp_values = self.get_exp_value(A_circuits)
        counter = 0
        for i in range(len(omega_t)):
            j = 0
            while j <= i:
                if j != i:
                    A[i,j] = fifj * np.real(A_exp_values[counter])
                    A[j,i] = A[i,j]
                    counter += 1
                j += 1
        return np.array(A)

    @abstractmethod
    def _get_dH_A_entry_circuit(self, i, j, s, omega_t, f_imag):
        r"""
        Get QuantumCircuit to compute the summand element of $dH_A_{i, j}$ w.r.t. the $s^{th}$ parameter of the Ansatz
        for the current circuit parameters $\omega_t$
        Args:
            i (int): index to denote the ith row of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            j (int): index to denote the ith column of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_A_{i, j}$ w.r.t. a Hamiltonian
            parameter $dH$
            omega_t (array-like): current circuit parameters
            f_imag (Bool): If true add an S gate to the evaluation QuantumRegister 'q' - This is needed if the summand
            coefficient $f^*_{k,i}$ for this term is imaginary.

        Returns:
            Quantum Circuit

        """
        return

    def get_dH_A(self, omega_t, dH_omega_t):
        r"""
        Get the tensor dH_A which represents the derivative of $A$ w.r.t. the Hamiltonian parameters
            i (int): index to denote the ith row of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            j (int): index to denote the ith column of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_A_{i, j}$ w.r.t. a Hamiltonian
            parameter $dH$
            omega_t (array-like): current circuit parameters

        Args:
            omega_t (array-like): current Ansatz parameters
            dH_omega_t: derivative of the current Ansatz parameters w.r.t. the Hamiltonian parameters

        Returns:
            nested array: dH_A

        """
        dH_A_circuits = []
        for s in range(len(omega_t)):
            for i in range(len(omega_t)):
                for j in range(len(omega_t)):
                    dH_A_circuits.append(self._get_dH_A_entry_circuit(i, j, s, omega_t, f_imag=True))

        dH_A_exp_values = self.get_exp_value(dH_A_circuits)
        counter = 0
        fifj = 1 / 8.
        dH_A = []
        for s in range(len(omega_t)):
            dH_A_temp = np.zeros((len(omega_t), len(omega_t)))
            for i in range(len(omega_t)):
                for j in range(len(omega_t)):
                    dH_A_temp[i, j] = fifj * np.real(dH_A_exp_values[counter])
                    counter += 1
            # We can exploit a symmetry in equation (A1) to reduce the number of circuit runs
            dH_A_temp = dH_A_temp+ np.transpose(dH_A_temp)
            dH_A.append(np.array(dH_A_temp))
        # tensor of dim kxkxk
        # dH_A = np.stack(dH_A, axis=0)
        #dH_omega_t: kxj
        #dH_A: kxkxk
        dH_A = np.tensordot(np.transpose(dH_omega_t), dH_A, axes=1)
        #dH_A: jxkxk
        return np.array(dH_A)

    @abstractmethod
    def _get_C_entry_circuit(self, i, pauli, omega_t, f_imag):
        r"""
        Get QuantumCircuit to compute the summand vector element $C_{i}$ w.r.t. the given pauli term for the current
        circuit parameters $\omega_t$
        Args:
            i (int): index to denote the ith element of C
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters
            f_imag (Bool): If true add an S gate to the evaluation QuantumRegister 'q' - This is needed if the summand
            coefficient $f^*_{k,i}$ in $C_i$ of Eq. (6) in [Variational ansatz-based quantum simulation of imaginary
            time evolution](https://doi.org/10.1038/s41534-019-0187-2) is imaginary.

        Returns:
            Quantum Circuit
        Raises:
            NotImplementedError

        """
        raise NotImplementedError

    def get_C(self, omega_t):
        r"""
        Get the matrix $C$ for the current circuit parameters $\omega_t$
        #k - num params in the VarForm used for SITE
        #j - num terms in the Hamiltonian = dim of the H params
        Args:
            omega_t (array-like): current circuit parameters

        Returns:
            array: $C$

        """
        fi = - 1 / 2.  # - comes from McLachlan's principle
        C = np.zeros((len(omega_t)))
        # Store outcomes for the computation of dH_C to reduce the number of circuit runs
        if len(self._C_circs) == 0:
            # self._C_circs: kxj
            self._C_circs = np.zeros((len(omega_t), len(self._hamiltonian.paulis)))
        C_circuits = []
        for i in range(len(omega_t)):
            for weight, pauli in self._hamiltonian.paulis:
                C_circuits.append(self._get_C_entry_circuit(i, pauli, omega_t, f_imag=True))

        counter = 0
        C_exp_values = self.get_exp_value(C_circuits)
        for i in range(len(omega_t)):
            j = 0
            for weight, pauli in self._hamiltonian.paulis:
                C[i] += fi * weight * np.real(C_exp_values[counter])
                try:
                    self._C_circs[i, j] = fi * np.real(C_exp_values[counter])
                except Exception:
                    continue
                counter += 1
                j += 1
        return np.array(C)

    @abstractmethod
    def _get_dH_C_entry_circuit1(self,  j, s, pauli, omega_t, f_imag):
        r"""
        Get the first quantum circuit to compute the summand element of $dH_C_{j}$ w.r.t. the $s^{th}$ parameter of the
        Ansatz and for one pauli term of the target Hamiltonian of the imaginary time evolution for the current circuit
        parameters $\omega_t$ by evaluating the expectation value $/langle Z /rangle$ w.r.t.
        the QuantumRegister 'q'
        Args:
            j (int): index to denote the jth entry of $dH_C$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_C_{j}$ w.r.t. a Hamiltonian
            parameter $dH$
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters

        Returns:
            Quantum Circuit

        """
        return

    @abstractmethod
    def _get_dH_C_entry_circuit2(self, j, s, pauli, omega_t):
        r"""
        Get the second quantum circuit to compute the summand element of $dH_C_{j}$ w.r.t. the $s^{th}$ parameter of the
        Ansatz and for one pauli term of the target Hamiltonian of the imaginary time evolution for the current circuit
        parameters $\omega_t$ by evaluating the expectation value $/langle Z /rangle$ w.r.t.
        the QuantumRegister 'q'
        Args:
            j (int): index to denote the jth entry of $dH_C$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_C_{j}$ w.r.t. a Hamiltonian
            parameter $dH$
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters

        Returns:
            Quantum Circuit

        """
        # This operator is adjoint for j, s
        return

    def get_dH_C(self, omega_t, dH_omega_t):
        r"""
        Get the tensor dH_C which represents the derivative of $C$ w.r.t. the Hamiltonian parameters
        Args:
            omega_t (array-like): current Ansatz parameters
            dH_omega_t: derivative of the current Ansatz parameters w.r.t. the Hamiltonian parameters

        Returns:
            array: $dH_C$

        """
        if len(self._C_circs) == 0:
            self.get_C(omega_t)
        else:
            pass
        fsfj = 1 / 4.  # (-1)*(-1) McLachlan
        dH_C_temp = np.zeros((len(omega_t), len(omega_t)))
        dH_C_circuits = []

        for s in range(len(omega_t)):
            for j in range(len(omega_t)):
                for weight, pauli in self._hamiltonian.paulis:
                    dH_C_circuits.append(self._get_dH_C_entry_circuit1(j, s, pauli, omega_t))
                    dH_C_circuits.append(self._get_dH_C_entry_circuit2(j, s, pauli, omega_t))

        dH_C_exp_val = self.get_exp_value(dH_C_circuits)
        counter = 0
        for s in range(len(omega_t)):
            for j in range(len(omega_t)):
                for weight, pauli in self._hamiltonian.paulis:
                    dH_C_temp[j, s] += weight * fsfj * np.real(dH_C_exp_val[counter])
                    counter += 1
                    dH_C_temp[j, s] -= weight * fsfj * np.real(dH_C_exp_val[counter])
                    counter += 1
        #dH_omega_t: kxj
        #dH_C_temp: kxk
        dH_C_temp = np.tensordot(np.real(dH_C_temp), dH_omega_t, axes=1)
        # dH_C: kxj
        #self._C_circs already includes the neccessary factors
        # - from McLachlan already included in self._C_cics
        # print('self C circs ', np.linalg.norm(self._C_circs))
        dH_C_temp = dH_C_temp.reshape(np.shape(self._C_circs))
        dH_C = self._C_circs + dH_C_temp
        return dH_C

    def get_exp_value(self, circuit):
        r"""
        Evaluate the expectation value $\langle Z \rangle$ w.r.t. the ancilla qubit (named 'q')
        Args:
            circuit(QuantumCircuit or list of QuantumCircuits): with a single qubit QuantumRegister with name 'q'

        Returns: (list of) expectation value $\langle Z \rangle$ w.r.t. the 'q' QuantumRegister

        """

    @staticmethod
    def _get_A_entry_circuit(ansatz, i, j, omega_t, global_phase_fix=True):
        r"""
        Get QuantumCircuit to compute the matrix element $A_{i,j}$ for the current circuit parameters $\omega_t$
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            i (int): index to denote the ith row of A
            j (int): index to denote the jth column of A
            omega_t (array-like): current circuit parameters
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                        for a potential mismatch between the target and the trained state - see
                        `Theory of variational quantum simulation
                         <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            Quantum Circuit
        Raises:
            NotImplementedError

        """
        # phase_eval='+' i.e. initial state of the evaluation qubit |q> = |0> + |1>

    @staticmethod
    def get_A(ansatz, omega_t, global_phase_fix=True):
        r"""
        Get the matrix $A$ for the current circuit parameters $\omega_t$
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            omega_t (array-like): current circuit parameters
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                        for a potential mismatch between the target and the trained state - see
                        `Theory of variational quantum simulation
                        <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            array: $A$

        """

    @staticmethod
    def _get_dH_A_entry_circuit(ansatz, i, j, s, omega_t, global_phase_fix=True):
        r"""
        Get QuantumCircuit to compute the summand element of $dH_A_{i, j}$ w.r.t. the $s^{th}$ parameter of the Ansatz
        for the current circuit parameters $\omega_t$
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            i (int): index to denote the ith row of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            j (int): index to denote the ith column of $dH_A$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_A_{i, j}$ w.r.t. a Hamiltonian
            parameter $dH$
            omega_t (array-like): current circuit parameters
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                        for a potential mismatch between the target and the trained state - see
                        `Theory of variational quantum simulation
                         <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            Quantum Circuit

        """
        # phase_eval='+i' i.e. initial state of the evaluation qubit |q> = |0> + i|1>

    @staticmethod
    def get_dH_A(ansatz, omega_t, dH_omega_t, global_phase_fix=True):
        r"""
        Get the tensor dH_A which represents the derivative of $A$ w.r.t. the Hamiltonian parameters
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            omega_t (array-like): current Ansatz parameters
            dH_omega_t: derivative of the current Ansatz parameters w.r.t. the Hamiltonian parameters
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            nested array: dH_A

        """

    @staticmethod
    def _get_C_entry_circuit(ansatz, i, pauli, omega_t, phase_eval='+i', global_phase_fix=True):
        r"""
        Get QuantumCircuit to compute the summand vector element $C_{i}$ w.r.t. the given pauli term for the current
        circuit parameters $\omega_t$
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            i (int): index to denote the ith element of C
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters
            phase_eval ({'+', '+i'}): Phase of the initial state of the evaluation QuantumRegister 'q',
                                        i.e. |q> = |0> + phase_eval|1> ==> imaginary time '+i', real time '+'
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_


        Returns:
            Quantum Circuit
        Raises:
            NotImplementedError

        """

    @staticmethod
    def get_C(ansatz, omega_t, global_phase_fix=True):
        r"""
        Get the matrix $C$ for the current circuit parameters $\omega_t$
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            omega_t (array-like): current circuit parameters
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            array: $C$

        """

    @staticmethod
    def _get_dH_C_entry_circuit1(ansatz, j, s, pauli, omega_t, phase_eval='+', global_phase_fix=True):
        r"""
        Get the first quantum circuit to compute the summand element of $dH_C_{j}$ w.r.t. the $s^{th}$ parameter of the
        Ansatz and for one pauli term of the target Hamiltonian of the imaginary time evolution for the current circuit
        parameters $\omega_t$ by evaluating the expectation value $/langle Z /rangle$ w.r.t.
        the QuantumRegister 'q'
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            j (int): index to denote the jth entry of $dH_C$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_C_{j}$ w.r.t. a Hamiltonian
            parameter $dH$
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters
            phase_eval ({'+', '+i'}): Phase of the initial state of the evaluation QuantumRegister 'q',
                                        i.e. |q> = |0> + phase_eval|1> ==> imaginary time '+', real time '+i'
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            Quantum Circuit

        """

    @staticmethod
    def _get_dH_C_entry_circuit2(ansatz, j, s, pauli, omega_t, phase_eval='+', global_phase_fix=True):
        r"""
        Get the first quantum circuit to compute the summand element of $dH_C_{j}$ w.r.t. the $s^{th}$ parameter of the
        Ansatz and for one pauli term of the target Hamiltonian of the imaginary time evolution for the current circuit
        parameters $\omega_t$ by evaluating the expectation value $/langle Z /rangle$ w.r.t.
        the QuantumRegister 'q'
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            j (int): index to denote the jth entry of $dH_C$ w.r.t. a Hamiltonian parameter $dH$
            s (int): index to denote the summand for the $s^{th}$Ansatz parameter of $dH_C_{j}$ w.r.t. a Hamiltonian
            parameter $dH$
            pauli (Pauli): Pauli term from the Hamiltonian for which we want to approximate the imaginary time evolution
            omega_t (array-like): current circuit parameters
            phase_eval ({'+', '+i'}): Phase of the initial state of the evaluation QuantumRegister 'q',
                                        i.e. |q> = |0> + phase_eval|1> ==> imaginary time '+', real time '+i'
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_

        Returns:
            Quantum Circuit

        """

    @staticmethod
    def get_dH_C(ansatz, omega_t, dH_omega_t, global_phase_fix=True):
        r"""
        Get the tensor dH_C which represents the derivative of $C$ w.r.t. the Hamiltonian parameters
        Args:
            ansatz (Ansatz): parametrized variational circuit (Ansatz) to implement the simulated imaginary time
                                     propagation
            omega_t (array-like): current Ansatz parameters
            dH_omega_t: derivative of the current Ansatz parameters w.r.t. the Hamiltonian parameters
                        global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                for a potential mismatch between the target and the trained state - see
                `Theory of variational quantum simulation
                 <https://quantum-journal.org/papers/q-2019-10-07-191/>`_
        Returns:
            array: dH_C

        """


class VarQTEResult(VarQTEResult):
    """ IQPE Result."""

    @property
    def phase(self) -> float:
        """ Returns phase """
        return self.get('phase')

    @phase.setter
    def phase(self, value: float) -> None:
        """ Sets phase """
        self.data['phase'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'IQPEResult':
        """ create new object from a dictionary """
        return IQPEResult(a_dict)