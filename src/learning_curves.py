import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from src.project_paths import RESULTS_DIR, TRAIN_PATH

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_train_data():
    df = pd.read_csv(TRAIN_PATH)
    label_col = "label"
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y

def evaluate_model(X_train, X_test, y_train, y_test, hidden_layers):
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=30,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1

def main():
    X, y = load_train_data()

    # Train sizes (percentage of available training data)
    train_sizes = [0.2, 0.4, 0.6, 0.8, 0.95]
    x = np.arange(1, len(train_sizes) + 1)
    x_labels = [f"{int(ts * 100)}%" for ts in train_sizes]


    # Containers for metrics
    acc_existing, acc_proposed = [], []
    prec_existing, prec_proposed = [], []
    rec_existing, rec_proposed = [], []
    f1_existing, f1_proposed = [], []

    # Loop over training sizes
    for ts in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ts, stratify=y, random_state=42
        )

        # Existing: simpler model
        acc_e, prec_e, rec_e, f1_e = evaluate_model(
            X_train, X_test, y_train, y_test, hidden_layers=(64,)
        )
        # Proposed: stronger model
        acc_p, prec_p, rec_p, f1_p = evaluate_model(
            X_train, X_test, y_train, y_test, hidden_layers=(128, 64)
        )

        acc_existing.append(acc_e)
        acc_proposed.append(acc_p)
        prec_existing.append(prec_e)
        prec_proposed.append(prec_p)
        rec_existing.append(rec_e)
        rec_proposed.append(rec_p)
        f1_existing.append(f1_e)
        f1_proposed.append(f1_p)

    # ---------- Plot helper ----------
    def plot_curves(y_exist, y_prop, ylabel, title, filename, color_exist, color_prop):
        plt.figure(figsize=(6, 4))
        plt.plot(x, y_exist, marker="o", linestyle="--", color=color_exist, label="Existing")
        plt.plot(x, y_prop, marker="s", linestyle="-", color=color_prop, label="Proposed")
        plt.xticks(x, x_labels)
        plt.ylim(0, 1.0)
        plt.xlabel("Training size")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename)
        plt.show()
        plt.close()

    # 1) Accuracy curve (often gently saturating)
    plot_curves(
        acc_existing,
        acc_proposed,
        "Accuracy",
        "Accuracy vs Training Size",
        "curve_accuracy_train_size.png",
        color_exist="#ff7f0e",
        color_prop="#1f77b4",
    )

    # 2) Precision curve (use different colors/shape)
    plot_curves(
        prec_existing,
        prec_proposed,
        "Precision (Macro)",
        "Precision vs Training Size",
        "curve_precision_train_size.png",
        color_exist="#d62728",
        color_prop="#2ca02c",
    )

    # 3) Recall curve
    plot_curves(
        rec_existing,
        rec_proposed,
        "Recall (Macro)",
        "Recall vs Training Size",
        "curve_recall_train_size.png",
        color_exist="#9467bd",
        color_prop="#8c564b",
    )

    # 4) F1 curve
    plot_curves(
        f1_existing,
        f1_proposed,
        "F1-Score (Macro)",
        "F1-Score vs Training Size",
        "curve_f1_train_size.png",
        color_exist="#e377c2",
        color_prop="#17becf",
    )

    print("Learning-curve graphs saved in results/ and displayed on screen.")

if __name__ == "__main__":
    main()
