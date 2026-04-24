#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
from scipy.optimize import minimize
from typing import Sequence

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator


def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(
        list(graph.nodes())
    ), "The length of x must coincide with the number of nodes in the graph."
    return sum(
        x[u] * (1 - x[v]) + x[v] * (1 - x[u])
        for u, v in list(graph.edge_list())
    )

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, list[int], float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)

    return cost

def plot_result(G, x):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos, _default_axes = rx.spring_layout(G), plt.axes(frameon=True)
    rx.visualization.mpl_draw(
        G, node_color=colors, node_size=100, alpha=0.8, pos=pos
    )



def main():
    n = 22
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    edge_list = [
    (0, 1, 1.0),
    (0, 3, 1.0),
    (0, 4, 1.41),
    (1, 2, 1.0),
    (1, 4, 1.0),
    (1, 3, 1.41),
    (1, 5, 1.41),
    (2, 5, 1.0),
    (2, 4, 1.41),
    (3, 4, 1.0),
    (3, 19, 2.0),
    (4, 5, 1.0),
    (4, 13, 2.0),
    (5, 6, 2.0),
    (16, 15, 1.0),
    (16, 14, 1.0),
    (16, 13, 1.41),
    (15, 11, 1.0),
    (15, 13, 1.0),
    (15, 14, 1.41),
    (15, 12, 1.41),
    (11, 12, 1.0),
    (11, 13, 1.41),
    (11, 10, 2.0),
    (14, 13, 1.0),
    (14, 18, 2.0),
    (13, 12, 1.0),
    (12, 8, 2.24),
    (17, 18, 1.41),
    (17, 19, 1.41),
    (18, 19, 2.0),
    (8, 9, 1.0),
    (8, 6, 1.0),
    (8, 7, 1.41),
    (9, 7, 1.0),
    (9, 10, 2.24),
    (6, 7, 1.0),
    ]

    graph.add_edges_from(edge_list)
    draw_graph(graph, node_size=600, with_labels=True)

# In[4]:





    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)
    print("Cost Function Hamiltonian:", cost_hamiltonian)

# #### Hamiltonian → quantum circuit
# 
# 

# In[5]:


    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    circuit.measure_all()

    circuit.draw("mpl")

    # In[6]:


    circuit.parameters

    # ### Step 2: Optimize problem for quantum hardware execution
    # 
    # 

    # In[7]:


    service = QiskitRuntimeService()
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=127
    )
    print(backend)

    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    candidate_circuit = pm.run(circuit)
    candidate_circuit.draw("mpl", fold=False, idle_wires=False)

    # In[18]:



    initial_gamma = 0.7025
    initial_beta = 1.9635
    init_params = [ initial_beta, initial_beta,  initial_gamma,  initial_gamma]

    # In[8]:




    # In[19]:



    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)

    objective_func_vals = []  # Global variable

    # No Session — just instantiate Estimator directly with the backend
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000
    # Note: twirling is supported on real hardware without a session
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    # Note: dynamical decoupling may not be supported outside a session on all backends
    # remove the lines below if you get an error
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)

    # In[ ]:



    objective_func_vals = []

    estimator = Estimator()
    estimator.options.default_shots = 1

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)

    # The optimizer was able to reduce the cost and find better parameters for the circuit.
    # 
    # 

    # In[20]:


    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()
    print(objective_func_vals)

    # In[22]:




    optimized_circuit = candidate_circuit.assign_parameters(result.x)
    optimized_circuit.draw("mpl", fold=False, idle_wires=False)

    # In[23]:


    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 2_000_000

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(2e5))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    print(final_distribution_int)

    # ### Step 4: Post-process and return result in desired classical format
    # 
    # The post-processing step interprets the sampling output to return a solution for your original problem. In this case, you are interested in the bitstring with the highest probability as this determines the optimal cut. The symmetries in the problem allow for four possible solutions, and the sampling process will return one of them with a slightly higher probability, but you can see in the plotted distribution below that four of the bitstrings are distinctively more likely than the rest.
    # 
    # 

    # In[24]:


    # auxiliary functions to sample most likely bitstring
    def to_bitstring(integer, num_bits):
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]


    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    print(values[np.argmax(np.abs(values))])

    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, len(graph))
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)

    # In[35]:


    matplotlib.rcParams.update({"font.size": 10})

    final_bits = final_distribution_bin

    # Convert once
    keys = list(final_bits.keys())
    vals = np.array(list(final_bits.values()))

    # Find indices of top 4 probabilities
    top_idx = np.argsort(vals)[-4:]

    # Build colors array once
    colors = np.full(len(vals), "tab:grey", dtype=object)
    colors[top_idx] = "tab:purple"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(vals)), vals, color=colors, width=1.0)

    ax.set_title("Result Distribution")
    ax.set_xlabel("Bitstring index")
    ax.set_ylabel("Probability")

    # No x tick labels
    ax.set_xticks([])

    plt.show()

    # #### Visualize best cut
    # 
    # From the optimal bit string, you can then visualize this cut on the original graph.
    # 
    # 

    # In[36]:


    # auxiliary function to plot graphs

    plot_result(graph, most_likely_bitstring)

    # And calculate the value of the cut:
    # 
    # 

    # In[37]:


    cut_value = evaluate_sample(most_likely_bitstring, graph)
    print("The value of the cut is:", cut_value)

if __name__ == "__main__":
    main()
