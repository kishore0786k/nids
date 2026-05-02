import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.project_paths import MODEL_DIR, ROBUST_MODEL_PATH, TRAIN_PATH


class FGSMAdversarialTrainer:
    def __init__(self, epsilon=0.01, epochs=5):
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_robust_model(self, X_train, y_train, X_test, y_test):
        print("Training robust MLP baseline for adversarial comparison.")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        joblib.dump(model, ROBUST_MODEL_PATH)
        return model


if __name__ == "__main__":
    print("Training adversarial-robust NIDS baseline.")
    train_df = pd.read_csv(TRAIN_PATH)
    X = train_df.drop(columns=["label"]).values
    y = pd.factorize(train_df["label"])[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = FGSMAdversarialTrainer()
    trainer.train_robust_model(X_train, y_train, X_test, y_test)
    print(f"Saved: {ROBUST_MODEL_PATH}")
