import pandas as pd
from build_graph import build_elliptic_graph
from classical_signals import compute_fan_ratios, detect_communities

G = build_elliptic_graph(
    "data/raw/elliptic/txs_features.csv",
    "data/raw/elliptic/txs_edgelist.csv",
    "data/raw/elliptic/txs_classes.csv",
)

# Use a slightly bigger slice — first 5 timesteps — for a more meaningful sample
nodes_sample = [n for n, d in G.nodes(data=True) if d.get("timestep") in [1, 2, 3, 4, 5]]
subG = G.subgraph(nodes_sample).copy()
print(f"Sample graph — Nodes: {subG.number_of_nodes()}, Edges: {subG.number_of_edges()}")

ratios = compute_fan_ratios(subG)
partition = detect_communities(subG)

rows = []
for node, data in subG.nodes(data=True):
    label = data.get("label")
    rows.append({
        "node": node,
        "label": label,  # '1'=illicit, '2'=licit, 'unknown'
        "fan_in": ratios[node]["fan_in"],
        "fan_out": ratios[node]["fan_out"],
        "pass_through_ratio": ratios[node]["pass_through_ratio"],
        "community": partition[node],
    })

df = pd.DataFrame(rows)
df["label"] = df["label"].astype(str)

# Drop unknowns — we only care about confirmed illicit vs licit for this check
labeled = df[df["label"].isin(["1", "2"])]

print("\n--- Mean pass-through ratio by label ---")
print(labeled.groupby("label")["pass_through_ratio"].mean())

print("\n--- Community size distribution: illicit concentration ---")
illicit_by_community = labeled[labeled["label"] == "1"].groupby("community").size().sort_values(ascending=False)
total_by_community = labeled.groupby("community").size()
concentration = (illicit_by_community / total_by_community).dropna().sort_values(ascending=False)
print(concentration.head(10))