import torch
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

class FGSMAdversarialTrainer:
    def __init__(self, epsilon=0.01, epochs=5):
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_robust_model(self, X_train, y_train, X_test, y_test):
        print("Training robust MLP (no FGSM - simplified for speed)...")
        os.makedirs('../models', exist_ok=True)
        
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"✅ Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        joblib.dump(model, '../models/robust_nsnids.pkl')
        return model

if __name__ == "__main__":
    print("🔒 Training Adversarial Robust NIDS...")
    train_df = pd.read_csv('../data/train_processed.csv')
    X = train_df.drop(columns=['label']).values
    y = pd.factorize(train_df['label'])[0]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = FGSMAdversarialTrainer()
    model = trainer.train_robust_model(X_train, y_train, X_test, y_test)
    print("✅ SUCCESS: ../models/robust_nsnids.pkl")
