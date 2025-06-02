import torch
from models.gnn_net import GNN

n = 5               # 5×5 board
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = GNN(n=n, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)
