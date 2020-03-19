## Pseudo-Code VarQTE


### VarSITE Ansatz Class
"""
This algorithm employs the parameter training of a given Ansatz w.r.t. a given Hamiltonian by using McLachlan`s
variational principle.
"""

class VarSITE(QuantumAlgorithm):
    """
    Variational Simulation of Imaginary Time Evolution
    arXiv:1804.03023

    """

    def __init__(self, regularization=None, measurement_error_mitigation=False):
        """
        Initialize Var SITE Algorithm
        Args:
            regularization(None or str: {'ridge', 'lasso', 'perturb_diag'}):
                if None no regularization applied in _get_dH_dt_weightsdt_weights but small perturbations are
                applied on the left side of the SLE and the solution is computed with lstsq,
                else use ridge or lasso with automatic optimal parameter search
                measurement_error_mitigation (Bool): Use measurement error mitigation (deprecated for statevector
                simulator) - for further information see
                `Qiskit Tutorials <https://github.com/Qiskit/qiskit-iqx-tutorials/>`_
        """


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

    def run(self, quantum_instance, imaginary_hamiltonian=None, real_hamiltonian=None, tau=1, num_time_steps=500,
            ansatz=None, initial_state=None, omega_0=None, q_target=None, ODE_Solver=None, for_qrbm=False,
            snapshot_dir=None, resume=False, global_phase_fix=False):
        r"""
        Run Var SITE Algorithm
        Args:
            quantum_instance (QuantumInstance): backend configuration instance
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
            for_qrbm (Boolean): True - VarSITE is used for QRBM state preparation, False - else
            snapshot_dir (Union(str, None)): path or None, if path given store cvs file
                                      with parameters to the directory
            resume(Bool): if True and snapshot_dir given, load params from snapshot_dir and resume algorithm
            global_phase_fix (Bool): if True add ancilla and apply X and a U1 gate with an add. parameter to compensate
                                    for a potential mismatch between the target and the trained state - see
                                    `Theory of variational quantum simulation
                                     <https://quantum-journal.org/papers/q-2019-10-07-191/>`_
        """

    def _run(self):
        """
        Algorithm is executed and the parameters are updated

        """
