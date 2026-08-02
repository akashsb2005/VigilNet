import torch
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
import xgboost as xgb
from splits import temporal_split

data = torch.load("../data/processed/elliptic_pyg.pt", weights_only=False)
train_mask, val_mask, test_mask = temporal_split(data)

X_train, y_train = data.x[train_mask].numpy(), data.y[train_mask].numpy()
X_test, y_test = data.x[test_mask].numpy(), data.y[test_mask].numpy()

model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), eval_metric="aucpr")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("=== XGBoost Baseline (temporal split) ===")
print(classification_report(y_test, preds, target_names=["licit", "illicit"]))
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, probs):.4f}")