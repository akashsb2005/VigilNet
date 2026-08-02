import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import torch
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
from gnn import FraudGAT

data = torch.load("../data/processed/elliptic_pyg.pt", weights_only=False)
model = FraudGAT(in_channels=data.x.shape[1])
model.load_state_dict(torch.load("../models/gat_best.pt"))
model.eval()

features_df = pd.read_csv("../data/raw/elliptic/txs_features.csv", low_memory=False, usecols=[0])
idx_to_txid = {i: int(row) for i, row in enumerate(features_df.iloc[:, 0])}

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
)

def explain_flagged_node(node_index):
    explanation = explainer(data.x, data.edge_index, index=node_index)
    top_edges = explanation.edge_mask.topk(min(5, explanation.edge_mask.numel()))
    important_neighbors = data.edge_index[:, top_edges.indices][0].tolist()
    top_features = explanation.node_mask[node_index].topk(5).indices.tolist()
    import torch.nn.functional as F
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index)[node_index], dim=0)
    return {
        "node_index": node_index,
        "real_txid": idx_to_txid[node_index],
        "confidence": probs[1].item(),
        "important_neighbor_indices": important_neighbors,
        "top_feature_indices": top_features,
    }

if __name__ == "__main__":
    flagged_nodes = (data.y == 1).nonzero(as_tuple=True)[0][:3].tolist()
    for n in flagged_nodes:
        print(explain_flagged_node(n))