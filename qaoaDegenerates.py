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

GRAPHS = True
minimizeParamaters = False
maxDegenerates = ["01001010010110101010","10110101101001010101","10010101101001010101","01101010010110101010"]
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

    #transpile the circuit such that it has a hardware config, add the cost hamiltonian to it
    candidate_circuit = transpile(circuit, backend=backend, optimization_level=3)
    isa_hamiltonian = cost_hamiltonian.apply_layout(candidate_circuit.layout)

    #These were found with pain
    initial_gamma = 0.7025
    initial_beta = 1.9635
    init_params = [ initial_beta, initial_beta,  initial_gamma,  initial_gamma]

    objective_func_vals = []

    estimator = Estimator()
    estimator.options.default_shots = 8192
    if minimizeParamaters:
        result = minimize(
            cost_func_estimator,
            init_params,
            args=(candidate_circuit, isa_hamiltonian, estimator, objective_func_vals),
            method="COBYLA",
            tol=1e-2,
        )
        print("PARAMETERS:")
        print(result.x)
        optimized_circuit = candidate_circuit.assign_parameters(result.x)
    else:
        print("Used past paramaters")
        params = [2.77517392, 2.93571047, 0.23707659, 0.55289918]
        optimized_circuit = candidate_circuit.assign_parameters(params)

    optimized_circuit.measure_all()

    optimized_circuit = transpile(optimized_circuit, backend=backend, optimization_level=3)
    valuePerShot = [[],[],[],[]]
    percentOfStates = []
    shotArray = np.linspace(10_000,3_000_000,600)
    for final_shots in shotArray:
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
        percentOfStates.append(len(final_distribution_bin)/2**n * 100)
        for i in np.arange(0,4):
            valuePerShot[i].append(final_distribution_bin[maxDegenerates[i]]*100)
    for i, arr in enumerate(valuePerShot):
        plt.plot(shotArray,arr,label=f"{maxDegenerates[i]}")
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Num Of Shots")
    plt.ylabel("Probabiltiy(%)")
    plt.savefig("individualProb.png",dpi=800)

    plt.show()

    sumArray = np.sum(valuePerShot, axis=0)
    plt.plot(shotArray, sumArray, label="Sum of degenerate states",color='darkblue')
    plt.xlabel("Num of Shots")
    plt.ylabel("Probability (%)")
    plt.xscale('log')
    plt.legend()
    plt.savefig("totalProb.png",dpi=800)
    plt.show()

    avgArray = np.mean(valuePerShot, axis=0)
    plt.plot(shotArray, avgArray, label="Avg of degenerate states",color='darkorange')
    plt.xlabel("Num of Shots")
    plt.ylabel("Probability (%)")
    plt.xscale('log')
    plt.legend()
    plt.savefig("meanProb.png",dpi=800)
    plt.show()

    plt.plot(shotArray,percentOfStates,label="Percent of accessed states",color='darkgreen')
    plt.ylabel("Percent of accessed states (%)")
    plt.xlabel("Number of Shots")
    plt.xscale('log')
    plt.legend()
    plt.savefig("statesPercent.png",dpi=800)
    plt.show()
if __name__ == "__main__":
    main()