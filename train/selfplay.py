# train/selfplay.py
import numpy as np
from env.game_env import game_env
from mcts.mcts import MCTS

def run_selfplay_one_game(net, n, mcts_params):
    """
    Plays one complete game using current net + MCTS.
    Returns: a list of (state, adj_edge_index, legal_mask, pi, current_player_at_state)
             for every move, along with final winner (+1, -1, or 0).
    """
    env = game_env(n=n)
    mcts = MCTS(net, **mcts_params)
    trajectory = []  # will store (state, edge_index, legal_mask, pi, player_id)

    while True:
        # 1) Get current observation
        state, legal_mask = env._get_observation()  # state: [E×3], legal_mask: [E]

        # 2) Run MCTS → (best_move, pi)
        action, pi = mcts.get_move(env)

        # 3) Record this state for later training
        # Deep‐copy the numpy arrays; clone the PyTorch tensor
        trajectory.append((
            state.copy(),
            env.adj_edge_index_tensor.clone(),
            legal_mask.copy(),
            pi.copy(),
            env.current_player
        ))

        # 4) Take the step
        _, reward, done, info = env.step(action)

        if done:
            # info == {"boxes": (a_score, b_score)}
            a_score, b_score = info["boxes"]  # <-- only unpack here
            if a_score > b_score:
                winner = +1
            elif a_score < b_score:
                winner = -1
            else:
                winner = 0  # in case of exact tie

            # Build training data: (state, edge_index, legal_mask, pi, z)
            data = []
            for (s, eidx, lmask, pi_vec, player) in trajectory:
                if winner == 0:
                    z = 0.0
                else:
                    # z = +1 if the player who moved at that state eventually won
                    z = +1.0 if player == winner else -1.0
                data.append((s, eidx, lmask, pi_vec, z))

            return data

        # If not done, continue looping

def generate_selfplay_data(net, n, num_games, mcts_params):
    buffer_data = []
    for _ in range(num_games):
        game_data = run_selfplay_one_game(net, n, mcts_params)
        buffer_data.extend(game_data)
    return buffer_data
