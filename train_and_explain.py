import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import seaborn as sns
import os

# Create output folder if not exists
os.makedirs("Outputs", exist_ok=True)

# ============================
# 1. Load Dataset
# ============================
print("Loading dataset...")
df = pd.read_excel(r"V:\VKfusion\pro1\file\default of credit card clients.xls", header=1)

# Rename target column
df.rename(columns={"default payment next month": "default_payment"}, inplace=True)

X = df.drop("default_payment", axis=1)
y = df["default_payment"]

# ============================
# 2. Train-test Split
# ============================
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 3. Train XGBoost Model
# ============================
print("Training XGBoost model...")
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# 4. Model Evaluation
# ============================
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("Outputs/confusion_matrix.png")
print("Saved: Outputs/confusion_matrix.png")

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.savefig("Outputs/roc_auc_curve.png")
print("Saved: Outputs/roc_auc_curve.png")

# ============================
# 5. SHAP Explainability
# ============================
print("Generating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values.values, X_test, show=False)
plt.savefig("Outputs/shap_summary_plot.png", bbox_inches="tight")
print("Saved: Outputs/shap_summary_plot.png")

# SHAP Feature Importance (bar)
plt.figure()
shap.summary_plot(shap_values.values, X_test, plot_type="bar", show=False)
plt.savefig("Outputs/shap_feature_importance.png", bbox_inches="tight")
print("Saved: Outputs/shap_feature_importance.png")

# SHAP Force Plot (Single Example)
print("Generating example SHAP force plot...")
sample_index = 10
shap.force_plot(
    explainer.expected_value,
    shap_values.values[sample_index],
    X_test.iloc[sample_index, :],
    matplotlib=True
)
plt.savefig("Outputs/shap_force_plot_sample.png", bbox_inches="tight")
print("Saved: Outputs/shap_force_plot_sample.png")

print("\nAll tasks completed! Check the Outputs/ folder.")