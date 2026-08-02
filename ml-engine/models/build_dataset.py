import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph_analytics'))
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from build_graph import build_elliptic_graph
from classical_signals import compute_fan_ratios, detect_communities

def build_pyg_data():
    G = build_elliptic_graph(
        "data/raw/elliptic/txs_features.csv",
        "data/raw/elliptic/txs_edgelist.csv",
        "data/raw/elliptic/txs_classes.csv",
    )
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    ratios = compute_fan_ratios(G)
    partition = detect_communities(G)

    features, labels, timesteps = [], [], []
    for n in nodes:
        d = G.nodes[n]
        base_feats = list(d["features"])
        engineered = [ratios[n]["fan_in"], ratios[n]["fan_out"], ratios[n]["pass_through_ratio"], partition[n]]
        features.append(base_feats + engineered)
        lbl = str(d.get("label"))
        labels.append(1 if lbl == "1" else (0 if lbl == "2" else -1))  # -1 = unknown
        timesteps.append(d["timestep"])

    # Clean + normalize features (fixes NaN/inf and wildly different scales)
    features_arr = np.array(features, dtype=np.float64)
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    features_arr = scaler.fit_transform(features_arr)

    x = torch.tensor(features_arr, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    ts = torch.tensor(timesteps, dtype=torch.long)

    edge_index = torch.tensor(
        [[node_idx[u], node_idx[v]] for u, v in G.edges()], dtype=torch.long
    ).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.timestep = ts
    return data

if __name__ == "__main__":
    data = build_pyg_data()
    print(data)
    torch.save(data, "data/processed/elliptic_pyg.pt")
    print("Saved to data/processed/elliptic_pyg.pt")