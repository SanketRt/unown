import numpy as np
from utils import *

class game_env:
    def __init__(self, n):
        self.n = n
        self.E = 2 * n * (n + 1)                
        self.adj_list = build_line_graph(n)  
        self.reset()

    def reset(self):
        # Initialize node features:
        #   f1 = zeros(E), f2 = zeros(E), f3 = zeros(E)
        #   current_player = +1  (we’ll call +1 “agent A”, –1 “agent B”)
        self.f1 = np.zeros(self.E, dtype=np.int8)
        self.f2 = np.zeros(self.E, dtype=np.int8)
        self.f3 = compute_all_slacks(self.f1, self.n)  # initially zero everywhere
        self.current_player = +1
        self.box_counts = {+1: 0, -1: 0}  # how many boxes each side has
        return self._get_observation()

    def _get_observation(self):
        # Returns:
        #   node_feats: shape (E, 3) stacked [f1; f2; f3]
        #   to_draw_mask: length‐E vector {0/1}, 1 if f1[i]==0
        return np.stack([self.f1, self.f2, self.f3], axis=1), (self.f1 == 0).astype(np.int8)

    def step(self, action):
        """
        action: integer in [0, E). If f1[action] != 0, illegal.
        Returns: (obs_next, reward, done, info)
        """
        if self.f1[action] != 0:
            raise ValueError("Illegal action selected")

        # Draw the edge:
        self.f1[action] = 1
        self.f2[action] = self.current_player

        # Check if this completes 1×1 box(es):
        completed_boxes = 0
        for cell in adjacent_boxes_of_edge(action, self.n):
            # Count how many edges around that box are now drawn:
            if count_edges_drawn(cell,self.f1,self.n) == 4:
                completed_boxes += 1

        # Update slack features: For each neighbor j ∈ adj_list[action]:
        # Recompute f3[j] based on updated f1 around that edge.
        for j in self.adj_list[action]:
            self.f3[j] = count_adjacent_three_sided_boxes(j, self.f1, self.n)

        # Assign reward and update box_counts:
        if completed_boxes > 0:
            reward = float(completed_boxes)  
            self.box_counts[self.current_player] += completed_boxes
        else:
            reward = 0.0
            self.current_player *= -1

        # Check for terminal (all edges drawn):
        done = (self.f1.sum() == self.E)
        if done:
            # Final reward: +1 for win, -1 for loss, 0 for tie
            a_score = self.box_counts[+1]
            b_score = self.box_counts[-1]
            if a_score > b_score:
                final_reward = +1.0 if self.current_player == +1 else -1.0
            elif a_score < b_score:
                final_reward = -1.0 if self.current_player == +1 else +1.0
            else:
                final_reward = 0.005
            return (self._get_observation(), final_reward, True, 
                    {"boxes": (a_score, b_score)})

        # If not done, return intermediate reward 0.0
        return (self._get_observation(), reward, False, {"boxes_gained": completed_boxes})
