import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.ticker as ticker
import math 


G = nx.Graph()
edge_list =[
    (0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5),
    (0, 4), (1, 3), (1, 5), (2, 4), (16, 15), (15, 11),
    (14, 13), (13, 12), (16, 14), (15, 13), (11, 12),
    (16, 13), (14, 15), (15, 12), (11, 13), (17, 18),
    (17, 19), (18, 19), (8, 9), (6, 7), (8, 6), (9, 7), (10, 9),
    (4, 13), (18, 14), (19, 3), (12, 8), (5, 6), (11, 10), (8, 7)
]
G.add_edges_from(edge_list)

pos = {
    0: (-1, 0), 1: (0, 0), 2: (1, 0), 3: (-1, 1), 4: (0, 1), 5: (1, 1), 
    6: (3, 1), 7: (4, 1), 8: (3, 2), 9: (4, 2), 10: (3, 4), 
    11: (1, 4), 12: (1, 3), 13: (0, 3), 14: (-1, 3), 15: (0, 4), 16: (-1, 4),
    17: (-4, 2), 18: (-3, 3), 19: (-3, 1)
}

# Create Graph and calculate weights based on distance
G = nx.Graph()
for u, v in edge_list:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    # Euclidean distance formula
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    G.add_edge(u, v, weight=round(distance, 2))

# Setup Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Draw Nodes and Edges
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=600, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)

# Draw Edge Labels (the weights)
edge_labels = nx.get_edge_attributes(G, 'weight')

print("[")
for edge in G.edges(data=True):
    print(f"({edge[0]}, {edge[1]}, {edge[2]['weight']}),")
print("]")

#UNCOMMENT THE LINE BELOW TO PRINT WITH EDGE LABELS
#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

# Grid and Axis Formatting
ax.set_axis_on()
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

plt.xlim(-5.5, 5.5)
plt.ylim(-1.5, 5.5)
plt.title("Graph with Grid-Distance Weights")

plt.show()
