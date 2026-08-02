import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, classification_report
from gnn import FraudGAT
from splits import temporal_split

data = torch.load("../data/processed/elliptic_pyg.pt", weights_only=False)
train_mask, val_mask, test_mask = temporal_split(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
model = FraudGAT(in_channels=data.x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Class weights for imbalance (Gap 1)
y_train = data.y[train_mask]
n_illicit, n_licit = (y_train == 1).sum().item(), (y_train == 0).sum().item()
class_weights = torch.tensor([1.0, n_licit / n_illicit], dtype=torch.float32).to(device)

best_val_ap = 0
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1].cpu().numpy()
            val_labels = data.y[val_mask].cpu().numpy()
            val_ap = average_precision_score(val_labels, val_probs)
            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val PR-AUC {val_ap:.4f}")
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                torch.save(model.state_dict(), "gat_best.pt")

model.load_state_dict(torch.load("gat_best.pt"))
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_probs = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
    test_labels = data.y[test_mask].cpu().numpy()
    test_preds = (test_probs > 0.5).astype(int)
    print("\n=== GAT Final Test Results (temporal holdout) ===")
    print(classification_report(test_labels, test_preds, target_names=["licit", "illicit"]))
    print(f"Test PR-AUC: {average_precision_score(test_labels, test_probs):.4f}")