# train/replay_buffer.py
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, adj_edge_index, legal_mask, pi, z):
        """
        state: numpy array [E×3]
        adj_edge_index: torch.LongTensor [2, num_edges_line]
        legal_mask: numpy array [E]
        pi: numpy array [E]   (MCTS visit distribution)
        z: float (+1 or -1)
        """
        self.buffer.append((state, adj_edge_index, legal_mask, pi, z))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Unpack into tensors:
        states, edge_idxs, legal_masks, pis, zs = zip(*batch)
        return states, edge_idxs, legal_masks, pis, zs

    def __len__(self):
        return len(self.buffer)
