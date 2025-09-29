from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from utils import *


#Algorithm for Teleporting N qubits, using N + 1 qubit GHZ State

def tlprtn_algo_GHZ(circ, q, n, psi_nq):

    circ.initialize(psi_nq, q[:n])

    circ.barrier()

    CX_StateSelec(circ, n)

    circ.barrier()

    nGHZ(circ, n)

    circ.barrier()

    superpstn(circ, n)

    circ.barrier()

    meas_tlprtn(circ, n)

    circ.barrier()

    circ.reset(range(n + 1))



def teleportation_circuit_GHZ(n, psi_nq):

    q = QuantumRegister(2 * n + 1, 'q')
    c = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(q, c)

    tlprtn_algo_GHZ(qc, q, n, psi_nq)

    return qc, c



def tlprtn_algo_QFT(circ, q, n, psi_nq):

    circ.initialize(psi_nq, q[:n])

    circ.barrier()

    circ.append(QFT(2 * n, inverse = False), q[::-1])

    circ.barrier()

    circ.append(QFT(n, inverse=True), q[:n][::-1])

    circ.append(QFT(n, inverse=True), q[n:][::-1])



def teleportation_circuit_QFT(n, psi_nq):

    q = QuantumRegister(2 * n, 'q')
    c = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(q, c)

    tlprtn_algo_QFT(qc, q, n, psi_nq)

    return qc, c



#Circuit with state as the one to teleport to check for fidelity

def checker_algo(circ, q, n, psi_nq):

    circ.reset(range(n + 1))

    circ.barrier()
    
    circ.initialize(psi_nq, q[n + 1:])


def checker_circuit(n, psi_nq):

    q = QuantumRegister(2 * n + 1, 'q')
    c = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(q, c)

    checker_algo(qc, q, n, psi_nq)

    return qc, c



def checker_algo_QFT(circ, q, n, psi_nq):

    #circ.barrier()
    
    circ.initialize(psi_nq, q[n:])


def checker_circuit_QFT(n, psi_nq):

    q = QuantumRegister(2 * n, 'q')
    c = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(q, c)

    checker_algo_QFT(qc, q, n, psi_nq)

    return qc, c