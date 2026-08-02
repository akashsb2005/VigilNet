import sys, os
ML_ENGINE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-engine')
sys.path.append(os.path.join(ML_ENGINE_PATH, 'models'))
sys.path.append(os.path.join(ML_ENGINE_PATH, 'explain'))

import torch
import torch.nn.functional as F
import pandas as pd
from gnn import FraudGAT

_data = torch.load(os.path.join(ML_ENGINE_PATH, 'data/processed/elliptic_pyg.pt'), weights_only=False)
_model = FraudGAT(in_channels=_data.x.shape[1])
_model.load_state_dict(torch.load(os.path.join(ML_ENGINE_PATH, 'models/gat_best.pt')))
_model.eval()

import json
with open(os.path.join(ML_ENGINE_PATH, 'data/processed/elliptic_txid_order.json')) as f:
    _txids = json.load(f)
_txid_to_idx = {txid: i for i, txid in enumerate(_txids)}
_idx_to_txid = {v: k for k, v in _txid_to_idx.items()}

with torch.no_grad():
    _all_probs = F.softmax(_model(_data.x, _data.edge_index), dim=1)[:, 1]


def score_by_txid(txid: int):
    if txid not in _txid_to_idx:
        return None
    idx = _txid_to_idx[txid]
    confidence = _all_probs[idx].item()
    return {"node_index": idx, "txid": txid, "confidence": confidence}


def list_sample_transactions(limit: int = 25):
    top_idx = _all_probs.topk(min(limit, len(_all_probs))).indices.tolist()
    return [
        {"txid": _idx_to_txid[i], "confidence": _all_probs[i].item()}
        for i in top_idx
    ]


def search_transactions(query: str, limit: int = 10):
    matches = [txid for txid in _txid_to_idx.keys() if query in str(txid)][:limit]
    return [
        {"txid": t, "confidence": _all_probs[_txid_to_idx[t]].item()}
        for t in matches
    ]


def score_new_transaction(features: dict):
    """Score a transaction never seen during training, using engineered features only.

    Limitation, stated honestly: a brand-new transaction has no real edges yet, so it's
    scored as an isolated node appended to the existing graph. GNNs rely heavily on
    neighborhood structure, so this score reflects feature-based risk only, not full
    network-context risk. A production system would need incremental graph construction
    to give this a real neighborhood before scoring.
    """
    num_original_features = _data.x.shape[1] - 4  # last 4 are engineered: fan_in, fan_out, ratio, community
    values = [0.0] * num_original_features
    values += [
        features.get("fan_in", 0.0),
        features.get("fan_out", 0.0),
        features.get("pass_through_ratio", 0.0),
        features.get("community_id", 0.0),
    ]
    x_new = torch.tensor([values], dtype=torch.float32)

    combined_x = torch.cat([_data.x, x_new], dim=0)
    new_idx = combined_x.shape[0] - 1

    with torch.no_grad():
        out = _model(combined_x, _data.edge_index)
        prob = F.softmax(out[new_idx], dim=0)[1].item()

    return {"confidence": prob, "note": "Scored on engineered features only; no graph neighborhood available for a new transaction."}