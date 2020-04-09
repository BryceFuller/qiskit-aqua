import numpy as np
import scipy.linalg
import math
import netket as nk
import sys
import json
from functools import reduce
import random

###########################################################
# SINGLE QUBIT OPERATORS

# Identity Matrix
I = np.asarray([[1.,0.],[0.,1.]])

# Pauli X Matrix
X = np.asarray([[0.,1.],[1.,0.]])

# Pauli Y Matrix
Y = np.asarray([[0.,-1j],[1j,0.]])

# Pauli Z Matrix
Z = np.asarray([[1.,0.],[0.,-1.]])

# Unitary rotation into the Pauli X basis
rotationX = 1./(math.sqrt(2))*np.asarray([[1.,1.],[1.,-1.]])

# Unitary rotation into the Pauli Y basis
rotationY= 1./(math.sqrt(2))*np.asarray([[1.,-1j],[1.,1j]])

###########################################################

def OperatorFromString(op_string):
    '''
        Create a many-body operator over the full Hilbert space
        ---------------------------------------------------
        Inputs:
        op_string       list of char
                        ex: ['I','Z','Z','I']
        ---------------------------------------------------
        Outputs:
        operator        2-D array
                        ex: I x Z x Z x I 
    '''

    op_list = []
    for k in range(len(op_string)):
        if (op_string[k] == 'X'):
            op_list.append(X)
        elif (op_string[k] == 'Y'):
            op_list.append(Y)
        elif (op_string[k] == 'Z'):
            op_list.append(Z)
        else:
            op_list.append(I)
    operator = reduce(np.kron,op_list)
    return operator

###########################################################

def LocalOperatorFromString(op_string):
    '''
        Create a many-body operator over the full Hilbert space
        ---------------------------------------------------
        Inputs:
        op_string       list of char
                        ex: ['I','Z','Z','I']
        ---------------------------------------------------
        Outputs:
        sites           list of integers
                        ex: [1,2]
        operator        2-D array
                        ex: Z x Z
    '''
    op_list = []
    sites = []
    for k in range(len(op_string)):
        # If 'X', attach a sigmaX operator
        if (op_string[k] == 'X'):
            op_list.append(X)
            sites.append(k)
        # If 'Y', attach a sigmaY operator
        elif (op_string[k] == 'Y'):
            op_list.append(Y)
            sites.append(k)
        # If 'Z', attach a sigmaZ operator
        elif (op_string[k] == 'Z'):
            op_list.append(Z)
            sites.append(k)
    operator = reduce(np.kron,op_list)
    return sites,operator

###########################################################

def GenerateHamiltonian(hilbert,pauli,interactions):
    '''
        Generate model Hamiltonian
        ---------------------------------------------------
        Inputs:
        hilbert         netket.hilbert
                        Hilbert space
        pauli           list of pauli operators
                        [P_1,P_2,...] P_j=['I','Z','Z','I']
        interactions    list of floats
                        [c_1,c_2,...] 
        ---------------------------------------------------
        Outputs:
        hamiltonian     netket.operator
                        Hamiltonian
    '''
    N = hilbert.size    # Number of sites
    hamiltonian = nk.operator.LocalOperator(hilbert, 0.0)   # Initializer
    
    # Loop over the Pauli operators
    for h in range(len(pauli)):
        # Check if pauli is the identity
        identity = 'I'*N
        if(''.join(pauli[h]) ==  identity):
            # if identity: add constant energy shift
            hamiltonian += nk.operator.LocalOperator(hilbert,np.real(interactions[h]))
        else:
            # Else, add a local operator
            sites,operator = LocalOperatorFromString(pauli[h])
            h_term = interactions[h]*operator
            hamiltonian += nk.operator.LocalOperator(hilbert,h_term,sites)
    
    return hamiltonian

###########################################################

def Overlap(state1,state2):
    '''
        Calculate the overlap between two quantum states
        ---------------------------------------------------
        Inputs:
        state1/state2   1-D or 2-D arrays
                        wavefunctions or density matrices
        ---------------------------------------------------
        Outputs:
        overlap         float
                        Overlap

    '''
    if (len(state1.shape) ==1):
        # state1 is pure
        if (len(state2.shape) ==1):
            # state2 is pure
            overlap = np.abs(np.vdot(state1,state2))
        else:
            # state2 is mixed
            overlap = np.sqrt(np.abs(np.dot(np.conjugate(state1), np.dot(state2, state1)))) 
    else:
        #state1 is mixed
        if (len(state2.shape) == 1):
            # state2 is pure
            overlap = np.sqrt(np.abs(np.vdot(state2, np.dot(state1, state2))))
        else:
            # state2 is mixed
            sqrt_1 = scipy.linalg.sqrtm(state1)
            tmp = np.dot(sqrt_1,np.dot(state2,sqrt_1))
            overlap = np.trace(scipy.linalg.sqrtm(tmp)).real
    return overlap

###########################################################

def SampleBasis(N,basis_id,pauli=None):
    '''
        Sample a random basis according to basis_id
        ---------------------------------------------------
        Inputs:
        N               int
                        Number of qubits
        basis_id        string
                        Sampling criterion
        pauli           list of pauli operators
                        Used to sample in the Hamiltonian 
                        measurement basis
        ---------------------------------------------------
        Outputs:
        basis           list of chars
                        comment

    '''

    # Standard basis: ZZZZZZ...
    if (basis_id == 'std'):
        basis = ['Z' for j in range(N)]
    
    # Single X or Y (the rest is Z)
    if (basis_id == '1xy'):
        index = np.random.randint(2*N+1)
        basis = ['Z' for j in range(N)]
        if ((index>0) and (index<N+1)):
            basis[index-1] = 'X'
        elif (index > N):
            basis[index-(N+1)] = 'Y'

    # Random basis
    if (basis_id == 'random'):
        basis = np.random.choice(list('XYZ'),size=N,p=[1/3.,1/3.,1/3.])
    
    # Hamiltonian basis
    if (basis_id == 'ham'):
        # Uniformly sample one pauli operator
        index = np.random.randint(len(pauli))
        basis = []
        for j in range(N):
            if (pauli[index][j] == 'I'):
                basis.append('Z')
            else:
                basis.append(pauli[index][j])
    return basis

# Unitary rotation of a wavefunction
def RotateWavefunction(hilbert,psi,basis):
    rotation = nk.operator.LocalOperator(hilbert, 1.0)
    N = hilbert.size
    assert(len(basis) == hilbert.size)

    for j in range(hilbert.size):
        if (basis[j] == 'X'):
            rotation *= nk.operator.LocalOperator(hilbert, rotationX, [j])
        if (basis[j] == 'Y'):
            rotation *= nk.operator.LocalOperator(hilbert, rotationY, [j])
    U = rotation.to_sparse()
    psir = U.dot(psi)
    return psir

# Unitary rotation of a density matrix
def RotateDensityMatrix(hilbert,rho,basis):
    rotation = nk.operator.LocalOperator(hilbert, 1.0)
    N = hilbert.size
    assert(len(basis) == hilbert.size)

    for j in range(hilbert.size):
        if (basis[j] == 'X'):
            rotation *= nk.operator.LocalOperator(hilbert, rotationX, [j])
        if (basis[j] == 'Y'):
            rotation *= nk.operator.LocalOperator(hilbert, rotationY, [j])
    U = rotation.to_dense()
    tmp  = U.dot(rho)
    rho_r = tmp.dot(np.conjugate(np.transpose(U)))
    return rho_r

# Generate training samples/bases
def GenerateTrainingData(hilbert,state,basis_id,nsamples,fout_samples,fout_bases,pauli=None):
    training_samples = []
    training_bases   = []
    N = hilbert.size

    for i in range(nsamples):
        basis = SampleBasis(N,basis_id,pauli)
        if (len(state.shape) == 1):
            psi_b = RotateWavefunction(hilbert,state,basis)
            prob = np.multiply(psi_b,np.conj(psi_b)).real
        else:
        #rho = np.outer(psi,np.conjugate(psi))
            rho_b = RotateDensityMatrix(hilbert,state,basis)
            prob = np.diag(rho_b) 
        index = np.random.choice(hilbert.n_states,size=1,p=prob)
        sample = hilbert.number_to_state(index)
        print(sample)
        for j in range(N):
            fout_samples.write("%d " % int(sample[j]))
            fout_bases.write("%s" % basis[j])
        fout_samples.write('\n')
        fout_bases.write('\n')
    
# Load data from file into correct format
def LoadData(hilbert, path_to_samples=None, path_to_bases=None, samples=None, bases=None):
    training_samples = []
    training_bases = []
    N = hilbert.size 
    
    if (samples == None) and (bases == None):

        assert(path_to_samples!=None), "Path to samples not provided"
        assert(path_to_bases!=None), "Path to bases not provided"

        
        samples = np.loadtxt(path_to_samples)
        assert(N == samples.shape[1])

        fin_bases = open(path_to_bases, 'r')
        lines = fin_bases.readlines()
        bases = []

        for b in lines:
            basis = ""
            assert(len(b) == N + 1)
            for j in range(N):
                basis += b[j]
            bases.append(basis)
        index_list = sorted(range(len(bases)), key=lambda k: bases[k])
    elif (samples == None) or (bases == None):
        raise ValueError("One but not both of samples/bases are None. Either pass in both or provide a path to the location of both.")
    else:
        assert(len(samples) == len(bases))
        #samples = np.array(samples)

    if isinstance(bases, np.ndarray):
        bases = bases.tolist()
    elif not isinstance(bases, list):
        assert ValueError("Bases must either be a list or ndarray")

    index_list = sorted(range(len(bases)), key=lambda k: bases[k])
    bases.sort()

    #print(bases[:10])
    print(samples[:10])
    print(type(samples))
    print(type(samples[0]))
    #print(index_list[:10])



    if isinstance(samples[0],np.ndarray):
        for i in range(len(samples)):
            training_samples.append(samples[index_list[i]].tolist())
    elif isinstance(samples, np.ndarray) and isinstance(samples[0],list):
        training_samples = samples.tolist()
    elif (isinstance(samples,list) and isinstance(samples[0],list)):
        training_samples = samples
    elif not (isinstance(samples,list) and isinstance(samples[0],list)):
        assert ValueError("Samples must either be a 2D list or 2D ndarray")

    rotations = []

    tmp = ''
    b_index = -1
    for b in bases:
        if (b != tmp):
            tmp = b
            localop = nk.operator.LocalOperator(hilbert, 1.0)

            for j in range(N):
                if (tmp[j] == 'X'):
                    localop *= nk.operator.LocalOperator(hilbert, rotationX, [j])
                if (tmp[j] == 'Y'):
                    localop *= nk.operator.LocalOperator(hilbert, rotationY, [j])

            rotations.append(localop)
            b_index += 1
        training_bases.append(b_index)

    #import ipdb; ipdb.set_trace()

    return tuple(rotations), np.asarray(training_samples), np.asarray(training_bases)

# Shuffle data
def ShuffleData(samples,bases):
    data_size = len(samples)
    order = np.arange(data_size)
    np.random.shuffle(order)
    new_bases = []
    new_samples = []
    for k in range(data_size):
        new_samples.append(samples[order[k]])
        new_bases.append(bases[order[k]])
    return np.asarray(new_samples),np.asarray(new_bases)

# Take a random subsamble of the data
def SliceData(samples,bases,nsamples):
    data_size = len(samples)
    new_bases = []
    new_samples = []
    for k in range(nsamples):
        index = random.randint(0,data_size-1)
        new_samples.append(samples[index])
        new_bases.append(bases[index])
    return np.asarray(new_samples),np.asarray(new_bases)

def LoadWavefunction(psi_path):
    # Load a wavefunction from file
    psi = np.loadtxt(psi_path).view(complex).reshape(-1) 
    return psi

def LoadDensityMatrix(rho_path):
    # Load a density matrix from file
    rho = np.loadtxt(rho_path)
    return rho



