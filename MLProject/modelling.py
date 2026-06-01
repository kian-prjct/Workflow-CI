"""
modelling.py (MLProject version)
==================================
Script training Wine Quality yang kompatibel dengan MLflow Project.
Mendukung parameter via CLI (argparse) untuk integrasi dengan MLProject.

Usage (via MLflow):
    mlflow run . -P n_estimators=200 -P max_depth=10

Usage (via Python):
    python modelling.py --n_estimators 200 --max_depth 10
"""

import os
import argparse
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

warnings.filterwarnings('ignore')


# ── Konfigurasi DagsHub ────────────────────────────────────────────────────
DAGSHUB_OWNER   = os.getenv("DAGSHUB_OWNER", "kian-prjct")
DAGSHUB_REPO    = os.getenv("DAGSHUB_REPO",  "Workflow-CI")
EXPERIMENT_NAME = "wine-quality-ci-pipeline"

# ── Konfigurasi Data ───────────────────────────────────────────────────────
DATA_DIR   = "winequality_preprocessing"
TRAIN_PATH = os.path.join(DATA_DIR, "winequality_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "winequality_test.csv")
TARGET_COL = "quality_label"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Wine Quality Classifier")
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=int,   default=10)
    parser.add_argument("--min_samples_split", type=int,   default=5)
    parser.add_argument("--min_samples_leaf",  type=int,   default=2)
    parser.add_argument("--test_size",         type=float, default=0.2)
    parser.add_argument("--random_state",      type=int,   default=42)
    return parser.parse_args()


def load_data():
    """Memuat dataset train dan test."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL]
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Membuat confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Bad (0)', 'Good (1)']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, save_path):
    """Membuat feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    """Membuat ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print("  CI TRAINING: Wine Quality (MLflow Project)")
    print(f"{'='*55}")

    # ── Setup MLflow — simpan lokal di CI ─────────────────────────────────
    print("🔗 Setup MLflow tracking...")
    import os
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Buat experiment jika belum ada
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Load data ──────────────────────────────────────────────────────────
    print("📂 Memuat dataset...")
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()

    with mlflow.start_run(run_name="ci_training_run"):

        # ── Train model ────────────────────────────────────────────────────
        print("🚀 Melatih model...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # ── Evaluasi ───────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        metrics = {
            "accuracy":   accuracy_score(y_test, y_pred),
            "precision":  precision_score(y_test, y_pred),
            "recall":     recall_score(y_test, y_pred),
            "f1_score":   f1_score(y_test, y_pred),
            "roc_auc":    roc_auc_score(y_test, y_prob),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std":  cv_scores.std(),
        }

        # ── Manual Logging ─────────────────────────────────────────────────
        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
        mlflow.log_param("random_state",      args.random_state)

        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        # ── Log model ──────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wine-quality-ci"
        )

        # ── Log artefak tambahan ───────────────────────────────────────────
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, "plots")

        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
        plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path, "plots")

        roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
        plot_roc_curve(y_test, y_prob, metrics["roc_auc"], roc_path)
        mlflow.log_artifact(roc_path, "plots")

        cr_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
        report = classification_report(y_test, y_pred,
                                       target_names=['bad', 'good'])
        with open(cr_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(cr_path, "reports")

        # ── Simpan model lokal untuk Docker build ─────────────────────────
        os.makedirs("model_output", exist_ok=True)
        mlflow.sklearn.save_model(model, "model_output/wine_quality_model")
        print(f"\n  Model lokal disimpan di: model_output/wine_quality_model")

        run_id = mlflow.active_run().info.run_id
        print(f"\n📊 Metrik:")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")
        print(f"\n✅ Run ID: {run_id}")
        print(f"✅ Training selesai!")


if __name__ == "__main__":
    main()
