#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
from scipy.optimize import minimize
from typing import Sequence

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import QasmSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator

GRAPHS = False

def evaluate_sample(division: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(division) == len(
        list(graph.nodes())
    ), "The length of x must coincide with the number of nodes in the graph."

    total: float = 0
    for u, v in graph.edge_list():
        if division[u] != division[v]:
            total += graph.get_all_edge_data(u, v)[0] 

    return total
    
def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, list[int], float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

def cost_func_estimator(params, ansatz, isa_hamiltonian, estimator, objective_func_vals):
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)

    return cost


    # auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def main():
    #Construct the Graph
    n = 20
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    edge_list = [
    (0, 1, 1.0),
    (0, 3, 1.0),
    (0, 4, 1/1.41),
    (1, 2, 1.0),
    (1, 4, 1.0),
    (1, 3, 1/1.41),
    (1, 5, 1/1.41),
    (2, 5, 1.0),
    (2, 4, 1/1.41),
    (3, 4, 1.0),
    (3, 19, 1/2.0),
    (4, 5, 1.0),
    (4, 13, 1/2.0),
    (5, 6, 1/2.0),
    (16, 15, 1.0),
    (16, 14, 1.0),
    (16, 13, 1/1.41),
    (15, 11, 1.0),
    (15, 13, 1.0),
    (15, 14, 1/1.41),
    (15, 12, 1/1.41),
    (11, 12, 1.0),
    (11, 13, 1/1.41),
    (11, 10, 1/2.0),
    (14, 13, 1.0),
    (14, 18, 1/2.0),
    (13, 12, 1.0),
    (12, 8, 1/2.24),
    (17, 18, 1/1.41),
    (17, 19, 1/1.41),
    (18, 19, 1/2.0),
    (8, 9, 1.0),
    (8, 6, 1.0),
    (8, 7, 1/1.41),
    (9, 7, 1.0),
    (9, 10, 1/2.24),
    (6, 7, 1.0),
    ]
    graph.add_edges_from(edge_list)

    #Make the Hamiltonian from max cut paulis 
    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)

    #Make the Ansatz with the cost_hamiltonian as the, uh, cost 
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    backend = QasmSimulator()

    candidate_circuit = transpile(circuit, backend=backend, optimization_level=3)
    isa_hamiltonian = cost_hamiltonian.apply_layout(candidate_circuit.layout)

    initial_gamma = 0.7025
    initial_beta = 1.9635
    init_params = [ initial_beta, initial_beta,  initial_gamma,  initial_gamma]

    objective_func_vals = []

    estimator = Estimator()
    estimator.options.default_shots = 8192

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, isa_hamiltonian, estimator, objective_func_vals),
        method="COBYLA",
        tol=1e-2,
    )

    optimized_circuit = candidate_circuit.assign_parameters(result.x)
 
    optimized_circuit.measure_all()

    optimized_circuit = transpile(optimized_circuit, backend=backend, optimization_level=3)

    final_shots = 250_000
    job = backend.run(optimized_circuit, shots=final_shots)
    raw_counts_bin = job.result().get_counts()
    
    normalized_counts_bin = {}
    for bitstring, count in raw_counts_bin.items():
        normalized_bitstring = bitstring.replace(" ", "")
        normalized_counts_bin[normalized_bitstring] = (
            normalized_counts_bin.get(normalized_bitstring, 0) + count
        )

    shots = sum(normalized_counts_bin.values()) #WHY???? its just shots?
    print(f"shots: {shots}")

    
    final_distribution_bin = {
        key: val / shots for key, val in normalized_counts_bin.items()
    }

    final_distribution_int = {
        int(key, 2): val / shots for key, val in normalized_counts_bin.items()
    }
    #print(final_distribution_int)



    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())
    print(values[np.argmax(np.abs(values))])

    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, len(graph))
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)

    

    top_k = 10

    # 1) Top by probability
    top_by_prob = sorted(
        final_distribution_bin.items(),
        key=lambda kv: kv[1],
        reverse=True
    )[:top_k]

    print(
        format(
            sorted(
            final_distribution_bin.items(),
            key=lambda kv: kv[1],
            reverse=True
            )[-1][1], ".12f"
        )
    )

    print("\nTop candidates by probability:")
    for rank, (bitstring, prob) in enumerate(top_by_prob, 1):
        bits = [int(ch) for ch in bitstring][::-1]
        score = evaluate_sample(bits, graph)
        print(f"{rank:2d}. p={prob:.6f}  score={score:.3f}  bits={bits}")

    # 2) Top by weighted score among all sampled candidates
    scored = []
    for bitstring, prob in final_distribution_bin.items():
        bits = [int(ch) for ch in bitstring][::-1]
        score = evaluate_sample(bits, graph)
        scored.append((score, prob, bits))

    top_by_score = sorted(scored, key=lambda t: t[0], reverse=True)[:top_k]

    print("\nTop candidates by weighted cut score:")
    for rank, (score, prob, bits) in enumerate(top_by_score, 1):
        print(f"{rank:2d}. score={score:.3f}  p={prob:.6f}  bits={bits}")
    
    cut_value = evaluate_sample(most_likely_bitstring, graph)
    print(f"The value of the cut is: {cut_value:.3f}")


'''
    if GRAPHS: circuit.draw("mpl")

    if GRAPHS: draw_graph(graph, node_size=600, with_labels=True)

    plt.figure(figsize=(11, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if GRAPHS: plt.show()
    #print(objective_func_vals)

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

    if GRAPHS: plt.show()

    if GRAPHS: optimized_circuit.draw("mpl", fold=False, idle_wires=False)
'''


if __name__ == "__main__":
    main()








