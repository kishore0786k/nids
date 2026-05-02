import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import json
import os

# paths relative to src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_processed.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test_processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ns_nids_model.pkl")
RESULTS_JSON = os.path.join(BASE_DIR, "results", "metrics.json")


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_col = "label"
    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]
    return X_train, X_test, y_train, y_test


def train_model():
    os.makedirs(os.path.join("..", "results"), exist_ok=True)
    os.makedirs(os.path.join("..", "models"), exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    classes = sorted(y_train.unique())

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=25,
        random_state=42,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # proposed metrics
    macro_avg = report.get("macro avg", {})
    proposed_precision = macro_avg.get("precision", 0.0)
    proposed_recall = macro_avg.get("recall", 0.0)
    proposed_f1 = macro_avg.get("f1-score", 0.0)
    proposed_accuracy = report.get("accuracy", 0.0)

    # fixed existing metrics
    existing_accuracy = 0.90
    existing_precision = 0.88
    existing_recall = 0.87
    existing_f1 = 0.875

    metrics_summary = {
        "classes": classes,
        "classification_report": report,
        "existing": {
            "accuracy": existing_accuracy,
            "precision_macro": existing_precision,
            "recall_macro": existing_recall,
            "f1_macro": existing_f1,
        },
        "proposed": {
            "accuracy": proposed_accuracy,
            "precision_macro": proposed_precision,
            "recall_macro": proposed_recall,
            "f1_macro": proposed_f1,
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    joblib.dump(mlp, MODEL_PATH)
    print("Training completed. Metrics saved to ../results/metrics.json and model to ../models/ns_nids_model.pkl.")


if __name__ == "__main__":
    train_model()
