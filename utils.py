import numpy as np
from itertools import product
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_ibm_runtime.fake_provider import FakeHanoiV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
import matplotlib.pyplot as plt



#Prepare random state

def state_prep(n, k = None):

    np.random.seed(k)
    vec = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
    psi_nq = vec / np.linalg.norm(vec)
    return psi_nq


#Select the entangled states

def CX_StateSelec(circ, n):

    for i in range(1, n):
        circ.cx([i - 1, i], [- n + i, - n + i])


#Create the N+1 qubit GHZ state

def nGHZ(circ, n):

    circ.h(n)
    for i in range(n, 2 * n):
        circ.cx(i, i + 1)


#Create superposition of all the states

def superpstn(circ, n):

    circ.cx(0, n)
    circ.h(range(n))


#Gates applied based on classical information

def meas_tlprtn(circ, n):

    for i in range(n):
        circ.cx(n, n + 1 + i)
        circ.cz(i, n + 1 + i)


#Reverse the order of qubits

def reverse_order(statevec):

    n = len(statevec.dims())
    tensor = statevec.data.reshape([2] * n)
    reversed_tensor = np.transpose(tensor, axes=range(n-1, -1, -1))
    reversed_flat = reversed_tensor.flatten()
    return reversed_flat


#State of required qubits from the whole statevector of 2n + 1 qubits

def extract_state(full_vec):

    n = int(((np.log(len(full_vec))/np.log(2)) - 1)//2)
    rev_sv = Statevector(reverse_order(full_vec), dims = full_vec.dims()[::-1])
    total_qubits=2 * n + 1
    wanted_qubits = range(n+1, 2 * n + 1)
    fixed_qubits_map={i: 0 for i in range(0, n + 1)}
    num_wanted = len(wanted_qubits)
    output_state = np.zeros(2 ** num_wanted, dtype=complex)

    for i in range(2 ** total_qubits):
        bin_str = format(i, f'0{total_qubits}b')
        keep = True
        for qb, val in fixed_qubits_map.items():
            if int(bin_str[qb]) != val:
                keep = False
                break
        if keep:
            # Extract sub-index for wanted qubits
            sub_idx = ''.join([bin_str[qb] for qb in wanted_qubits])
            output_state[int(sub_idx, 2)] += full_vec[i]

    norm = np.linalg.norm(output_state)
    if norm != 0:
        output_state /= norm

    return output_state


#If any state has 0 count

def fill_missing(data_dict, n_qubits):

    all_keys = [''.join(bits) for bits in product('01', repeat=n_qubits)]
    
    for key in all_keys:
        if key not in data_dict:
            data_dict[key] = 0

    return data_dict


#Measure required qubits

def measure_all(circs, cs, n):

    for circ, c in zip(circs, cs):
        circ.measure(range(n + 1, 2 * n + 1), c)


#Simulate a circuit on simulator/real backend

def simulate(circ, backend):

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(circ)

    #isa_circuit.draw("mpl", idle_wires=False)

    sampler = Sampler(backend)
    job = sampler.run([isa_qc], shots = 8192)

    return job


#Simulate a batch on real backend

def simulate_batch(circ, backend, max_circ):

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    mltpl_isa_qc = pm.run(circ)

    prtn_circ = []
    for i in range(0, len(circ), max_circ):
        prtn_circ.append(mltpl_isa_qc[i : i + max_circ])

    jobs = []

    with Batch(backend = backend):
        sampler = Sampler()
        for pc in prtn_circ:
            job = sampler.run(pc)
            jobs.append(job)

    return jobs


#Get result for a job

def get_result(job, n):

    pub_result = job.result()[0]
    counts = pub_result.data.c.get_counts()
    fill_missing(counts, n)

    return counts


#Get results for Batch

def get_batch_result(jobs, n):
    mltpl_counts = []

    for job in jobs:
        res = job.result()
        for pub in res:
            counts = pub.data.c.get_counts()
            fill_missing(counts, n)
            mltpl_counts.append(counts)

    return mltpl_counts


#Calculate fidelity with counts of each state

def calc_fdlty(counts, counts_):

    # Assume counts and counts_2 are the two measurement dictionaries
    all_keys = set(counts.keys()).union(counts_.keys())

    # Normalize to probability distributions
    total_1 = sum(counts.values())
    total_2 = sum(counts_.values())

    p1 = {k: counts.get(k, 0) / total_1 for k in all_keys}
    p2 = {k: counts_.get(k, 0) / total_2 for k in all_keys}

    # Compute classical fidelity
    fidelity = sum(np.sqrt(p1[k] * p2[k]) for k in all_keys) ** 2

    return fidelity


#Calculate fidelity with counts if each state in Batch Mode

def calc_batch_fdlty(mltpl_counts, mltpl_counts_):

    fdlty = []
    for counts, counts_ in zip(mltpl_counts, mltpl_counts_):
        fdlty.append(calc_fdlty(counts, counts_))
    
    return fdlty


#Calculate fidelity of states we got in Batch mode, after teleportation and the one we wanted to teleport

def calc_batch_statefdlty(mltpl_sv, mltpl_sv_):

    n = int(((np.log(len(mltpl_sv[0]))/np.log(2)) - 1)//2)
    statefdlty = []
    for sv, sv_ in zip(mltpl_sv, mltpl_sv_):
        statefdlty.append(state_fidelity(extract_state(sv), extract_state(sv_)))
    
    return statefdlty


#Plot the counts of a job

def plot_counts(counts, n):
    shots = [counts[key] for key in sorted(counts.keys())]
    count = shots
    labels = [''.join(bits) for bits in product('01', repeat=n)] # Corresponding labels for each bar

    # Create a larger figure to prevent clipping
    plt.figure(figsize=(16, 10))

    # Create the bar plot
    bars = plt.bar(labels, count, color='skyblue', edgecolor='black')

    # Add value labels on top of each bar (5 decimal precision)
    for i, count_ in enumerate(count):
        plt.text(i, count_ + max(count) * 0.02, f'{count_:.0f}', ha='center', va='bottom', fontsize=10)

    # Labeling and formatting
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.title('Measurement Outcome Counts from Quantum Computer')

    # Ensure layout fits everything
    plt.tight_layout()

    # Show plot
    plt.show()


#Plot the counts of jobs in Batch Mode

def plot_batch_counts(fdlty, state_fdlty):

    # Your data
    fdlty_quantum_computer = fdlty
    fidelityt_state = state_fdlty

    # X-axis: number of circuits
    num_circuits = list(range(1, len(fdlty_quantum_computer) + 1))  # [1, 2, 3, 4, 5]

    # Plotting
    plt.figure(figsize=(16, 10))
    plt.plot(num_circuits, fdlty_quantum_computer, color='red', marker='o', label='Fidelity on Quantum Computer')
    plt.plot(num_circuits, fidelityt_state, color='green', marker='s', label='Fidelity of State')

    # Set axis limits
    plt.ylim(0.8, 1.2)
    plt.xlabel("Circuit Number")
    plt.ylabel("Fidelity")
    plt.title("Fidelity Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show values above points
    for i, val in enumerate(fdlty_quantum_computer):
        plt.text(num_circuits[i], val + 0.02, f"{val:.5f}", ha='center', fontsize=9, color='red')

    for i, val in enumerate(fidelityt_state):
        plt.text(num_circuits[i], val + 0.02, f"{val:.5f}", ha='center', fontsize=9, color='green')

    plt.show()



