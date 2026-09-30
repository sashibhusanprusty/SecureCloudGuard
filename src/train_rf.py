from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def main() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = pd.read_csv(csv_path)

    df.columns = df.columns.map(lambda col: str(col).strip())
    df = df.replace([float("inf"), float("-inf")], float("nan")).dropna()

    df["target"] = df["Label"].astype(str).str.contains("DDoS", case=False).astype(int)

    X = df.drop(columns=["Label", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
    )
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=StratifiedKFold(n_splits=5),
        scoring="accuracy",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    train_acc = model.score(X_train, y_train)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    for fold_index, fold_accuracy in enumerate(cv_scores, start=1):
        print(f"Fold {fold_index} Accuracy: {fold_accuracy:.4f}")
    print(f"5-Fold CV Mean Accuracy: {cv_mean:.4f}")
    print(f"5-Fold CV Standard Deviation: {cv_std:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "rf_model.pkl")
    metrics = {
        "dataset_samples": int(len(y)),
        "training_accuracy": float(train_acc),
        "test_accuracy": float(acc),
        "cv_folds": [float(score) for score in cv_scores],
        "cv_mean_accuracy": float(cv_mean),
        "cv_std_accuracy": float(cv_std),
    }
    metrics_path = output_dir / "rf_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"\nModel saved to: {output_dir / 'rf_model.pkl'}")


if __name__ == "__main__":
    main()
