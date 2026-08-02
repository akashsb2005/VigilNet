from build_graph import build_elliptic_graph
from classical_signals import compute_fan_ratios, detect_cycles, detect_communities

G = build_elliptic_graph(
    "data/raw/elliptic/txs_features.csv",
    "data/raw/elliptic/txs_edgelist.csv",
    "data/raw/elliptic/txs_classes.csv",
)

# Filter to just timestep 1 for a fast smoke test
nodes_t1 = [n for n, d in G.nodes(data=True) if d.get("timestep") == 1]
subG = G.subgraph(nodes_t1).copy()
print(f"Subgraph (timestep 1) — Nodes: {subG.number_of_nodes()}, Edges: {subG.number_of_edges()}")

ratios = compute_fan_ratios(subG)
print("Sample fan ratios:", list(ratios.items())[:3])

cycles = detect_cycles(subG, max_length=6)
print(f"Cycles found: {len(cycles)}")

partition = detect_communities(subG)
print(f"Communities found: {len(set(partition.values()))}")