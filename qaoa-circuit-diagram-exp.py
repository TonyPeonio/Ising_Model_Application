#%%
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
import matplotlib.pyplot as plt

#%%
# --- Config ---
n = 4   # number of qubits (change freely)
p = 2   # number of QAOA layers

def make_U_gate(n_qubits, p):
    """
    Single U gate spanning all qubits, labelled with the full
    QAOA unitary equation for p layers.
    The label uses LaTeX — rendered correctly by Qiskit's mpl drawer.
    """
    # Terms ordered right-to-left (outermost layer first in label,
    # matching how the product is written mathematically)
    terms = []
    for layer in range(p, 0, -1):
        terms.append(rf"$e^{{-i\beta_{{{layer}}} H_M}}$")
        terms.append(rf"$e^{{-i\gamma_{{{layer}}} H_C}}$")

    # Join with newline so Qiskit stacks them in the gate box
    label = "\n".join(terms)

    gate = Gate(name="U", num_qubits=n_qubits, params=[])
    gate.label = label
    return gate

# --- Build circuit ---
qc = QuantumCircuit(n, n)

# Hadamard layer
qc.h(range(n))
qc.barrier()

# QAOA unitary U
U = make_U_gate(n, p)
qc.append(U, range(n))
qc.barrier()

# Measurements
qc.measure(range(n), range(n))

#%%
# --- Render ---
# 'mpl' drawer renders $...$ as proper LaTeX math
# Requires: pip install pylatexenc matplotlib
fig = qc.draw(
    output='mpl',
    style='clifford',  # options: 'clifford', 'iqx', 'bw', 'textbook'
    fold=-1,           # -1 = never fold, keeps everything on one line
    scale=1.0,
)

fig.savefig('qaoa_circuit.png', dpi=180, bbox_inches='tight')
print("Saved to qaoa_circuit.png")
plt.show()
# %%
