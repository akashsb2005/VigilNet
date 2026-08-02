import torch
import torch.nn.functional as F
import pandas as pd
from gnn import FraudGAT

data = torch.load("../data/processed/elliptic_pyg.pt", weights_only=False)
model = FraudGAT(in_channels=data.x.shape[1])
model.load_state_dict(torch.load("gat_best.pt"))
model.eval()

with torch.no_grad():
    probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]

features_df = pd.read_csv("../data/raw/elliptic/txs_features.csv", low_memory=False, usecols=[0])
idx_to_txid = {i: int(row) for i, row in enumerate(features_df.iloc[:, 0])}

top5_idx = probs.topk(5).indices.tolist()
for idx in top5_idx:
    print(f"txid={idx_to_txid[idx]}, confidence={probs[idx].item():.4f}")