import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, classification_report
from gnn import FraudGAT

data = torch.load("../data/processed/paysim_pyg.pt", weights_only=False)

# PaySim's 'step' ranges roughly 1-743 (hours); split by time, same principle as Elliptic
max_step = data.timestep.max().item()
train_cut, val_cut = int(max_step * 0.7), int(max_step * 0.85)
train_mask = data.timestep <= train_cut
val_mask = (data.timestep > train_cut) & (data.timestep <= val_cut)
test_mask = data.timestep > val_cut

model = FraudGAT(in_channels=data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

y_train = data.y[train_mask]
n_illicit, n_licit = (y_train == 1).sum().item(), (y_train == 0).sum().item()
class_weights = torch.tensor([1.0, max(n_licit, 1) / max(n_illicit, 1)], dtype=torch.float32)

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
            val_probs = F.softmax(out[val_mask], dim=1)[:, 1].numpy()
            val_ap = average_precision_score(data.y[val_mask].numpy(), val_probs)
            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Val PR-AUC {val_ap:.4f}")
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                torch.save(model.state_dict(), "gat_best_paysim.pt")

model.load_state_dict(torch.load("gat_best_paysim.pt"))
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_probs = F.softmax(out[test_mask], dim=1)[:, 1].numpy()
    test_preds = (test_probs > 0.5).astype(int)
    print("\n=== PaySim GAT Test Results ===")
    print(classification_report(data.y[test_mask].numpy(), test_preds, target_names=["licit", "illicit"]))
    print(f"Test PR-AUC: {average_precision_score(data.y[test_mask].numpy(), test_probs):.4f}")