import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.append('..')

class FGSMAdversarialTrainer:
    def __init__(self, epsilon=0.01, alpha=0.007, epochs=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fgsm_attack(self, model, X, y, criterion=nn.CrossEntropyLoss()):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        model.eval()
        
        loss = criterion(model(X_tensor), y_tensor)
        model.zero_grad()
        loss.backward()
        
        data_grad = X_tensor.grad.data
        perturbation = self.epsilon * data_grad.sign()
        adv_X = X_tensor + perturbation
        adv_X = torch.clamp(adv_X, X_tensor.min(), X_tensor.max())
        
        return adv_X.cpu().numpy()
    
    def train_robust_model(self, X_train, y_train, X_test, y_test):
        os.makedirs('../models', exist_ok=True)
        
        class TorchMLP(nn.Module):
            def __init__(self, input_size, hidden_sizes=[128, 64]):
                super().__init__()
                layers = []
                prev_size = input_size
                for h in hidden_sizes:
                    layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
                    prev_size = h
                layers.append(nn.Linear(prev_size, len(np.unique(y_train))))
                self.net = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.net(x)
        
        pytorch_model = TorchMLP(X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
        
        X_adv = X_train.copy()
        y_adv = y_train.copy()
        
        for epoch in range(self.epochs):
            print(f"Adversarial training epoch {epoch+1}/{self.epochs}")
            adv_samples = self.fgsm_attack(pytorch_model, X_train, y_train, criterion)
            
            mix_ratio = 0.2
            n_adv = int(len(X_train) * mix_ratio)
            adv_idx = np.random.choice(len(X_train), n_adv, replace=False)
            X_adv[adv_idx] = adv_samples[adv_idx]
        
        robust_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        robust_model.fit(X_adv, y_adv)
        
        train_acc = robust_model.score(X_train, y_train)
        test_acc = robust_model.score(X_test, y_test)
        print(f"✅ Robust Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        joblib.dump(robust_model, '../models/robust_nsnids.pkl')
        return robust_model

if __name__ == "__main__":
    print("🔒 Training Adversarial Robust NIDS...")
    train_df = pd.read_csv('../data/train_processed.csv')
    X = train_df.drop(columns=['label']).values
    y = pd.factorize(train_df['label'])[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = FGSMAdversarialTrainer()
    robust_model = trainer.train_robust_model(X_train, y_train, X_test, y_test)
    print("✅ Saved: ../models/robust_nsnids.pkl")
