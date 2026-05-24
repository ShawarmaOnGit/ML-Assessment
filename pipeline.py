import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score,)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import kagglehub

# ============================================
# 1. REUSABLE FUNCTIONS
# ============================================

def plot_confusion_matrix(cm, title):
    """
    Args:
    - cm: Confusion matrix array from sklearn
    - title: Title to display on the chart
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Predicted Legit", "Predicted Fraud"], yticklabels=["Actual Legit", "Actual Fraud"], ax=ax)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()

def evaluate_model(y_test, y_pred, y_proba, model_name):
    """
    Role: Print classification report and return metrics dict.

    Args:
    - y_test: True labels
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities for fraud class
    - model_name: Name of the model for display

    Returns: dict with model name, precision, recall, F1, ROC-AUC
    """
    print(f"{model_name}: Test Set Performance")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    tn, fp, fn, tp = cm.ravel()
    print(f"Fraud caught (TP): {tp}")
    print(f"Fraud missed (FN): {fn}")
    print(f"False alarms (FP): {fp}")
    print(f"Legit approved (TN):{tn}")

    return {
        "Model": model_name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


# ============================================
# 2. LOAD DATA
# ============================================

download_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = Path(download_path) / "creditcard.csv"

df = pd.read_csv(csv_path)

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")


# ============================================
# 3. PREPROCESSING
# ============================================

X = df.loc[:, df.columns != "Class"]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99, stratify=y,
)

print(f"Training set: {X_train.shape[0]:,} samples with {y_train.sum()} frauds")
print(f"Test set: {X_test.shape[0]:,} samples with {y_test.sum()} frauds")

# Scale Time and Amount only (V1-V28 are already PCA-scaled)
scaler = RobustScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

print("\nRobustScaler finished on Time and Amount.")
print(f"Amount median after scaling: {X_train['Amount'].median():.4f}")


# ============================================
# 4. CUSTOMER SEGMENTATION — K-MEANS
# ============================================

# silhouette score
cluster_numbers = list(range(2, 9))
silhouette_scores = []

for k in cluster_numbers:
    km = KMeans(n_clusters=k, n_init=10, random_state=99)
    labels = km.fit_predict(X_train)
    score  = silhouette_score(X_train, labels, sample_size=10_000, random_state=99)
    silhouette_scores.append(score)
    print(f"K={k} | Silhouette score: {score:.4f}")

best_k = cluster_numbers[silhouette_scores.index(max(silhouette_scores))]
print(f"\nBest K by silhouette score: {best_k}")

# Fit final K-Means model
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=99)
train_clusters = kmeans.fit_predict(X_train)
test_clusters  = kmeans.predict(X_test)


# Gain an understanding o the clusters
df_train = X_train.copy()
df_train["Cluster"] = train_clusters
df_train["Class"]   = y_train.values

profile = df_train.groupby("Cluster").agg(Transaction_Count=("Class", "count"), Fraud_Count=("Class", "sum"), Fraud_Rate=("Class", "mean"), Avg_Amount=("Amount", "mean")).round(3)

profile["Fraud_Rate"] = (profile["Fraud_Rate"] * 100).round(3)
print("\nCluster Profiles:")
print(profile.to_string())

# ============================================
# 5. HANDLE CLASS IMBALANCE USING SMOTE
# ============================================


before = Counter(y_train)
print(f"Before SMOTE: Legitimate={before[0]:,} | " f"Fraud={before[1]:,} | Ratio={before[0]//before[1]}:1")
smote = SMOTE(sampling_strategy=0.1, random_state=99)  # type: ignore
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)  # type: ignore
after = Counter(y_train_sm)
print(f"After SMOTE:  Legitimate={after[0]:,} | Fraud={after[1]:,} | Ratio={after[0]//after[1]}:1")
print(f"New training size: {len(X_train_sm):,}")

# ============================================
# 6. STAGE 1 — BASELINE MODELS
# ============================================


all_metrics = []
# 6.1 Logistic Regression
print("\nStarting Logistic Regression: \n")

lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=99)

lr.fit(X_train_sm, y_train_sm)
lr_pred  = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

lr_metrics = evaluate_model(y_test, lr_pred, lr_proba, "Logistic Regression")
all_metrics.append(lr_metrics)

# 6.2 Random Forest
print("\nStarting Random Forest: \n")

rf = RandomForestClassifier(class_weight="balanced", random_state=99, n_jobs=-1,)

rf.fit(X_train_sm, y_train_sm)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_model(y_test, rf_pred, rf_proba, "Random Forest")
all_metrics.append(rf_metrics)

# 6.3 XGBoost
print("\nStarting XGBoost: \n")

xgb_model = XGBClassifier(eval_metric="aucpr", random_state=99, n_jobs=-1, verbosity=0,)

xgb_model.fit(X_train_sm, y_train_sm)
xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_proba, "XGBoost")
all_metrics.append(xgb_metrics)

# ============================================
# 7. COST-SENSITIVE ANALYSIS — RANDOM FOREST
# ============================================

# Recover original transaction amounts from the scaled data
amounts_test = scaler.inverse_transform(X_test[["Time", "Amount"]])[:, 1]

thresholds = np.arange(0.05, 0.95, 0.01)
costs_rf   = []

for threshold in thresholds:
    y_pred_t = (rf_proba >= threshold).astype(int)
    missed_fraud      = (y_test == 1) & (y_pred_t == 0)
    missed_fraud_cost = amounts_test[missed_fraud].sum()

    # Legit flagged as fraud (costs $10 per investigation)
    false_alarms = (y_test == 0) & (y_pred_t == 1)
    false_alarm_cost = false_alarms.sum() * 10

    costs_rf.append({"threshold":  round(threshold, 2), "total_cost": missed_fraud_cost + false_alarm_cost})

df_costs_rf = pd.DataFrame(costs_rf)
optimal_rf  = df_costs_rf.loc[df_costs_rf["total_cost"].idxmin()]
default_rf = (df_costs_rf[df_costs_rf["threshold"] == 0.50]["total_cost"].values[0])

print(f"Optimal threshold: {optimal_rf['threshold']}")
print(f"Minimum total cost: ${optimal_rf['total_cost']:,.2f}")
print(f"Default (0.50) cost: ${default_rf:,.2f}")
print(f"Savings vs default: ${default_rf - optimal_rf['total_cost']:,.2f}")


# ============================================
# 8. STAGE 2 — MLP (PYTORCH)
# ============================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class FraudMLP(nn.Module):
    """
    Architecture: input (30 neurons) → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(1)
    Output is a raw logit and the sigmoid is applied inside BCEWithLogitsLoss.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        return x


model = FraudMLP(input_dim=X_train_sm.shape[1], hidden_dim1=64, hidden_dim2=32, output_dim=1).to(device)

print("\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare tensors
X_train_t = torch.from_numpy(X_train_sm.values).float().to(device)
y_train_t = (torch.from_numpy(y_train_sm.values).float().reshape(-1, 1).to(device))
X_test_t  = torch.from_numpy(X_test.values).float().to(device)
y_test_t  = torch.from_numpy(y_test.values).float().reshape(-1, 1).to(device)

loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

num_legit  = (y_train_sm == 0).sum()
num_fraud  = (y_train_sm == 1).sum()
pos_weight = torch.tensor([num_legit / num_fraud]).to(device)

loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training batches per epoch: {len(loader)}")

# Training loop
EPOCHS = 20
train_losses = []
test_losses = []

for epoch in range(EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_test_t), y_test_t).item()
        test_losses.append(test_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS}  |  Train Loss: {avg_train_loss:.4f}  |  Test Loss: {test_loss:.4f}")

print("Training Completed")

model.eval()
with torch.no_grad():
    mlp_proba = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()

mlp_pred = (mlp_proba > 0.5).astype(int)
mlp_metrics = evaluate_model(y_test, mlp_pred, mlp_proba, "MLP (PyTorch)")
all_metrics.append(mlp_metrics)

# ============================================
# 9. MODEL COMPARISON & RECOMMENDATION
# ============================================

df_results = pd.DataFrame(all_metrics).set_index("Model").round(4)

print("\nFinal Model Comparison:")
print(df_results.to_string())

best_model = df_results["F1-Score"].idxmax()

print(f"\nRecommended Model: {best_model}")
print(f"Precision: {df_results.loc[best_model, 'Precision']:.4f}")
print(f"Recall: {df_results.loc[best_model, 'Recall']:.4f}")
print(f"F1-Score: {df_results.loc[best_model, 'F1-Score']:.4f}")
print(f"ROC-AUC: {df_results.loc[best_model, 'ROC-AUC']:.4f}")