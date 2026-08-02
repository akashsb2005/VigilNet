import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph_analytics'))
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from build_graph_paysim import build_paysim_graph

TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def build_pyg_data():
    G, sample = build_paysim_graph("../data/raw/paysim/PS_20174392719_1491204439457_log.csv")
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    sent = sample.groupby("nameOrig").agg(
        total_sent=("amount", "sum"), n_sent=("amount", "count"), avg_sent=("amount", "mean")
    )
    received = sample.groupby("nameDest").agg(
        total_received=("amount", "sum"), n_received=("amount", "count"), avg_received=("amount", "mean")
    )
    type_counts = pd.get_dummies(sample[["nameOrig", "type"]], columns=["type"]).groupby("nameOrig").sum()

    # VECTORIZED timestep computation (this replaces the slow per-node full-dataframe scan)
    step_by_orig = sample.groupby("nameOrig")["step"].min()
    step_by_dest = sample.groupby("nameDest")["step"].min()

    nodes_df = pd.DataFrame({"node": nodes}).set_index("node")

    sent_r = sent.reindex(nodes_df.index).fillna(0)
    received_r = received.reindex(nodes_df.index).fillna(0)
    type_r = type_counts.reindex(nodes_df.index).fillna(0)
    step_orig_r = step_by_orig.reindex(nodes_df.index)
    step_dest_r = step_by_dest.reindex(nodes_df.index)
    timestep_r = pd.concat([step_orig_r, step_dest_r], axis=1).min(axis=1).fillna(0)

    type_cols = [f"type_{t}" for t in TYPE_CATEGORIES]
    for c in type_cols:
        if c not in type_r.columns:
            type_r[c] = 0

    features_arr = np.hstack([
        sent_r[["total_sent", "n_sent", "avg_sent"]].values,
        received_r[["total_received", "n_received", "avg_received"]].values,
        type_r[type_cols].values,
    ]).astype(np.float64)

    labels = np.array([G.nodes[n]["label"] for n in nodes], dtype=np.int64)
    timesteps = timestep_r.values.astype(np.int64)

    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    features_arr = StandardScaler().fit_transform(features_arr)

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
    torch.save(data, "../data/processed/paysim_pyg.pt")
    print("Saved to data/processed/paysim_pyg.pt")