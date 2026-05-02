import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

from src.project_paths import MODEL_DIR, TRAIN_PATH


class GNNZeroDayDetector(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.anomaly_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.anomaly_head(x))


class SimpleFlowGraphConverter:
    def __init__(self, max_nodes=50):
        self.max_nodes = max_nodes

    def flows_to_graphs(self, flows, window_size=20):
        graphs = []
        for i in range(0, len(flows), window_size):
            window = flows.iloc[i:i + window_size]
            node_features = np.zeros((self.max_nodes, 4))
            for j, (_, row) in enumerate(window.iterrows()):
                if j >= self.max_nodes:
                    break
                node_features[j] = [
                    row.get("flow_pkts_s", 0),
                    row.get("flow_bytes_s", 0),
                    row.get("flow_duration", 0),
                    1.0,
                ]
            edges = []
            for _ in range(100):
                u = np.random.randint(0, self.max_nodes)
                v = np.random.randint(0, self.max_nodes)
                if u != v:
                    edges.extend([u, v])
            graphs.append(Data(x=torch.FloatTensor(node_features), edge_index=torch.LongTensor(edges).view(2, -1)))
        return graphs


def train_gnn_simple(train_flows):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    converter = SimpleFlowGraphConverter()
    train_graphs = converter.flows_to_graphs(train_flows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNZeroDayDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for data in train_graphs[:50]:
            data = data.to(device)
            optimizer.zero_grad()
            anomaly_score = model(data.x, data.edge_index, data.batch)
            loss = anomaly_score.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / 50:.4f}")
    torch.save(model.state_dict(), MODEL_DIR / "gnn_zero_day.pth")
    joblib.dump(converter, MODEL_DIR / "gnn_scaler.pkl")
    print("GNN zero-day detector trained.")
    return model, converter


if __name__ == "__main__":
    print("Training GNN zero-day detector.")
    train_df = pd.read_csv(TRAIN_PATH)
    train_gnn_simple(train_df)
    print(f"Saved: {MODEL_DIR / 'gnn_zero_day.pth'}")
