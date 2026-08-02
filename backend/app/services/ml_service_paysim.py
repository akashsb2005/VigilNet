import os
import json
import torch
import torch.nn.functional as F
from gnn import FraudGAT

ML_ENGINE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-engine')
import sys
sys.path.append(os.path.join(ML_ENGINE_PATH, 'models'))

_data = torch.load(os.path.join(ML_ENGINE_PATH, 'data/processed/paysim_pyg.pt'), weights_only=False)
_model = FraudGAT(in_channels=_data.x.shape[1])
_model.load_state_dict(torch.load(os.path.join(ML_ENGINE_PATH, 'models/gat_best_paysim.pt')))
_model.eval()

with torch.no_grad():
    _all_probs = F.softmax(_model(_data.x, _data.edge_index), dim=1)[:, 1]

# Load the cached account-order mapping instead of rebuilding the graph from the raw CSV
with open(os.path.join(ML_ENGINE_PATH, 'data/processed/paysim_account_order.json')) as f:
    _nodes = json.load(f)

_account_to_idx = {n: i for i, n in enumerate(_nodes)}
_idx_to_account = {v: k for k, v in _account_to_idx.items()}


def score_by_account(account: str):
    if account not in _account_to_idx:
        return None
    idx = _account_to_idx[account]
    return {"account_index": idx, "account": account, "confidence": _all_probs[idx].item()}


def list_sample_paysim(limit: int = 25):
    top_idx = _all_probs.topk(min(limit, len(_all_probs))).indices.tolist()
    return [{"account": _idx_to_account[i], "confidence": _all_probs[i].item()} for i in top_idx]


def search_paysim_accounts(query: str, limit: int = 10):
    matches = [acc for acc in _account_to_idx.keys() if query.upper() in acc.upper()][:limit]
    return [{"account": a, "confidence": _all_probs[_account_to_idx[a]].item()} for a in matches]