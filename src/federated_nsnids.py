import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List
import copy

class FederatedNeuroSymbolicNIDS:
    def __init__(self, num_clients=5, rounds=10, client_fraction=0.8):
        self.num_clients = num_clients
        self.rounds = rounds
        self.client_fraction = client_fraction
        self.global_model = None
        self.client_models = []
        self.scaler = StandardScaler()
        
    def client_feature_selection(self, X_local: np.ndarray, top_k=10):
        """Chimp-optimized feature selection per client."""
        # Simple variance-based selection (replace with chimp opt later)
        variances = np.var(X_local, axis=0)
        top_features = np.argsort(variances)[-top_k:]
        return top_features
    
    def average_models(self, client_models: List[MLPClassifier]) -> MLPClassifier:
        """FedAvg: Average model weights."""
        global_model = copy.deepcopy(client_models[0])
        
        for layer in range(len(global_model.coefs_)):
            avg_coef = np.mean([m.coefs_[layer] for m in client_models], axis=0)
            global_model.coefs_[layer] = avg_coef
            
            if len(global_model.intercepts_) > layer:
                avg_intercept = np.mean([m.intercepts_[layer] for m in client_models], axis=0)
                global_model.intercepts_[layer] = avg_intercept
        
        return global_model
    
    def federated_train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Federated training simulation."""
        os.makedirs('models/clients', exist_ok=True)
        
        # Split data by simulated device types (add device_type column if needed)
        n_samples = len(X_train)
        client_data = []
        samples_per_client = n_samples // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else n_samples
            client_X = X_train[start_idx:end_idx]
            client_y = y_train[start_idx:end_idx]
            client_data.append((client_X, client_y))
        
        for round_num in range(self.rounds):
            print(f"Federated Round {round_num + 1}/{self.rounds}")
            selected_clients = np.random.choice(self.num_clients, 
                                              int(self.num_clients * self.client_fraction), 
                                              replace=False)
            
            client_models = []
            
            for client_id in selected_clients:
                X_client, y_client = client_data[client_id]
                
                # Local feature selection
                top_features = self.client_feature_selection(X_client)
                X_selected = X_client[:, top_features]
                
                # Local training
                local_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42+client_id)
                local_model.fit(X_selected, y_client)
                
                # Save client model
                joblib.dump(local_model, f'models/clients/client_{client_id}.pkl')
                client_models.append(local_model)
            
            # Global aggregation
            self.global_model = self.average_models(client_models)
        
        # Final global fine-tuning
        self.global_model.fit(X_train, y_train)
        joblib.dump(self.global_model, 'models/federated_nsnids.pkl')
        print("Federated training completed!")
        return self.global_model

if __name__ == "__main__":
    train_df = pd.read_csv('../data/train_processed.csv')
    X = train_df.drop(columns=['label']).values
    y = pd.factorize(train_df['label'])[0]
    
    fed_nids = FederatedNeuroSymbolicNIDS(num_clients=5, rounds=10)
    fed_nids.federated_train(X, y)
