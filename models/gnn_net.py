# models/gnn_net.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, n, hidden_dim=64):
        super().__init__()
        self.n = n
        self.E = 2 * n * (n + 1)
        self.hidden_dim = hidden_dim

        # Two GCN layers
        self.conv1 = GCNConv(in_channels=3, out_channels=hidden_dim)  
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)

        # Policy head: maps each node embedding -> one logit
        self.policy_head = nn.Linear(hidden_dim, 1)

        # Value head: maps pooled node embeddings -> scalar
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch=None, legal_mask=None):
        """
        x: [num_nodes, 3] node features (f1, f2, f3)
        edge_index: [2, num_edges_line] line graph adjacency
        batch: [num_nodes] batch-index if doing batched graphs (optional)
        legal_mask: [num_nodes] 0/1 indicating which edges are legal
        """
        # 1) GCN layers
        h = F.relu(self.conv1(x, edge_index))   # [E, hidden_dim]
        h = F.relu(self.conv2(h, edge_index))   # [E, hidden_dim]

        # 2) Policy logits (one per node)
        logits = self.policy_head(h).squeeze(dim=-1)  # [E]

        if legal_mask is not None:
            # Mask: set illegal moves to large negative so softmax=0
            illegal = (legal_mask == 0)
            logits = logits.masked_fill(illegal, float('-1e9'))

        # 3) Value: global mean pooling if batched, else just mean
        if batch is not None:
            m = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        else:
            m = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        v = torch.tanh(self.value_head(m)).squeeze(dim=-1)  # [batch_size] or scalar

        return logits, v
