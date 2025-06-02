# mcts/node.py
import math

class MCTSNode:
    def __init__(self, parent=None):
        self.parent = parent         # parent MCTSNode or None
        self.children = {}           # action -> child MCTSNode
        self.N = {}                  # action -> visit count
        self.W = {}                  # action -> total value
        self.P = {}                  # action -> prior prob (from GNN)
        self.Q = {}                  # action -> mean value = W/N
        self.is_expanded = False     # whether the network has been queried here
        self.player = None           # which player’s turn at this node

    def expand(self, prior_probs, player, legal_actions):
        """
        prior_probs: dict[action -> P(a|s)] for all a
        player: +1 or -1
        legal_actions: list of legal a
        """
        self.player = player
        for a in legal_actions:
            self.P[a] = prior_probs[a]
            self.N[a] = 0
            self.W[a] = 0.0
            self.Q[a] = 0.0
        self.is_expanded = True

    def select_child(self, c_puct):
        """
        Return (action, child_node) that maximizes:
          Q(s,a) + c_puct * P(s,a)*sqrt( sum_b N(b) ) / (1 + N(s,a) )
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        sum_N = sum(self.N[a] for a in self.N)
        for a in self.N:
            # UCB term
            u = c_puct * self.P[a] * math.sqrt(sum_N) / (1 + self.N[a])
            score = self.Q[a] + u
            if score > best_score:
                best_score = score
                best_action = a
                best_child = self.children.get(a, None)

        return best_action, best_child
