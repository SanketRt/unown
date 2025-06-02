# mcts/mcts.py
import copy
import math
import torch
import numpy as np
from mcts.node import MCTSNode

class MCTS:
    def __init__(self, net, c_puct=1.0, n_playout=400, device='cpu'):
        """
        net: the GNN (DotBoxGNN)
        c_puct: exploration constant
        n_playout: number of MCTS simulations per move
        device: 'cuda' or 'cpu'
        """
        self.net = net
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.device = device

    def _evaluate(self, env):
        """
        Given an environment (DotBEnv), extract:
         - node_features (E×3 tensor)
         - edge_index (adjacency for line‐graph)
         - legal_mask (E‐vector)
        Run the GNN to get (logits, value). Convert logits→probabilities.
        """
        x_np, legal_mask_np = env._get_observation()  # x_np: [E×3], legal_mask_np: [E]
        x = torch.tensor(x_np, dtype=torch.float, device=self.device)
        legal_mask = torch.tensor(legal_mask_np, dtype=torch.bool, device=self.device)
        # Build tiny batch of size=1:
        #   edge_index tensor stored somewhere globally, or pass in env if cached.
        edge_index = env.adj_edge_index_tensor.to(self.device)  # shape [2, num_edge_edges]
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(x, edge_index, batch=None, legal_mask=legal_mask)
            probs = torch.softmax(logits, dim=0).cpu().numpy()  # [E]
            v = v.item()  # scalar
        return probs, v

    def _playout(self, root_env, root_node):
        """
        Run one MCTS simulation from root_env (copy of environment) 
        and root_node. Mutates root_node’s statistics.
        """
        env = copy.deepcopy(root_env)
        node = root_node

        # 1) SELECTION / EXPANSION
        path = []
        while True:
            if not node.is_expanded:
                # At leaf: run policy/value network and expand
                probs, v = self._evaluate(env)
                legal = env._get_observation()[1].astype(bool)  # legal_mask
                legal_actions = [i for i, ok in enumerate(legal) if ok]
                # Build prior_probs dict only on legal
                prior = {a: float(probs[a]) for a in legal_actions}
                node.expand(prior_probs=prior, player=env.current_player, legal_actions=legal_actions)
                # STOP selection here; we got v
                leaf_value = v
                break

            # If already expanded, pick best child (PUCT)
            a, child = node.select_child(self.c_puct)
            if child is None:
                # Create new child node (unexpanded)
                child = MCTSNode(parent=node)
                node.children[a] = child
                # Simulate that action in env
                obs, reward, done, info = env.step(a)
                # If this move completed a box, env.current_player stays same;
                # else flips.  That is done inside env.step().
                # Because we reached a new child, we now leaf‐evaluate it next loop.
                node = child
                if done:
                    # Terminal state → leaf value is game outcome from current root’s pov.
                    a_score, b_score = info["boxes"]
                    if a_score > b_score:
                        leaf_value = +1.0  # root’s player perspective: root_node.player?
                    elif a_score < b_score:
                        leaf_value = -1.0
                    else:
                        leaf_value = 0.0
                    break
                continue

            # Otherwise, move to that child:
            obs, reward, done, info = env.step(a)
            # If this move completed a box, leaf “player” hasn’t changed (env handles).
            # But when we backprop, we will flip v‐sign if control changed.
            path.append((node, a))
            node = child

            if done:
                # Terminal leaf: set leaf_value from perspective of the root’s player
                a_score, b_score = info["boxes"]
                if a_score > b_score:
                    leaf_value = +1.0
                elif a_score < b_score:
                    leaf_value = -1.0
                else:
                    leaf_value = 0.0
                break

        # 2) BACKPROPAGATION
        # Walk path backwards, updating W, N, Q
        # We need to flip sign of leaf_value whenever the player at that node != leaf_player
        # But since we only know leaf_value from root’s vantage, we can just propagate:
        value_to_propagate = leaf_value
        for parent_node, action_taken in reversed(path):
            # If the player at parent_node != node.player, flip:
            if parent_node.player != env.current_player:
                # Actually, env.current_player is leaf’s current_player after last step.
                # But leaf_value is relative to **root**.  We want to flip at each turn switch.
                value_to_propagate = -value_to_propagate

            # Update stats for (parent_node, action_taken)
            parent_node.W[action_taken] += value_to_propagate
            parent_node.N[action_taken] += 1
            parent_node.Q[action_taken] = parent_node.W[action_taken] / parent_node.N[action_taken]

            # Move up
            env.current_player *= -1  # flip for next level up
            node = parent_node

    def get_move(self, root_env):
        """
        Given current environment, run self.n_playout playouts, then return:
          - move          = argmax_a (N(root, a))
          - pi_vector     = visit_count distribution over all a
        """
        # 1) Build root node
        root_node = MCTSNode(parent=None)
        # 2) Expand once (so we have a prior to start with)
        probs, v = self._evaluate(root_env)
        legal = root_env._get_observation()[1].astype(bool)
        legal_actions = [i for i, ok in enumerate(legal) if ok]
        prior = {a: float(probs[a]) for a in legal_actions}
        root_node.expand(prior_probs=prior, player=root_env.current_player, legal_actions=legal_actions)

        # 3) Run n_playout playouts
        for _ in range(self.n_playout):
            self._playout(root_env, root_node)

        # 4) Compute final move distribution π over all actions
        N_total = sum(root_node.N[a] for a in root_node.N)
        pi = np.zeros(root_env.E, dtype=np.float32)
        for a in root_node.N:
            pi[a] = root_node.N[a] / float(N_total)

        # 5) Choose move: argmax_a N[a]
        best_a = max(root_node.N.keys(), key=lambda a: root_node.N[a])
        return best_a, pi
