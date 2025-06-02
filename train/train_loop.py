# train/train_loop.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from replay_buffer import ReplayBuffer
from models.gnn_net import GNN
from selfplay import *
from env.game_env import game_env

class DotBShotDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list   # list of (state, edge_index, legal_mask, pi, z)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s, eidx, lmask, pi, z = self.data[idx]
        # Convert to tensors:
        s_t = torch.tensor(s, dtype=torch.float)       # [E×3]
        l_t = torch.tensor(lmask, dtype=torch.bool)     # [E]
        pi_t = torch.tensor(pi, dtype=torch.float)      # [E]
        z_t = torch.tensor(z, dtype=torch.float)        # scalar
        eidx_t = eidx.clone()                           # [2, num_edge_line]
        return s_t, eidx_t, l_t, pi_t, z_t

def train_network(net, optimizer, dataloader, device):
    net.train()
    total_loss = 0.0
    for s_batch, eidx_batch, lmask_batch, pi_batch, z_batch in dataloader:
        s_batch = s_batch.to(device)       # [B, E, 3]
        lmask_batch = lmask_batch.to(device) # [B, E]
        pi_batch = pi_batch.to(device)      # [B, E]
        z_batch = z_batch.to(device)        # [B]

        # Since graphs are all the same structure, we can combine them:
        # Flatten batch dimension into nodes dimension:
        B, E, _ = s_batch.shape
        x = s_batch.view(B * E, 3)    # [B*E, 3]
        edge_index = eidx_batch[0]    # they’re identical for each sample

        # We'll assign a batch vector: [0,0,0,...,1,1,1,..., B-1,B-1,B-1]
        # so global_mean_pool knows how to pool per‐graph.
        batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, E).view(-1)
        # Forward pass
        logits, values = net(x, edge_index.to(device), batch_idx, lmask_batch.view(-1))

        # Reshape:
        logits = logits.view(B, E)  # [B, E]
        values = values.view(B)     # [B]

        # 1) Policy loss: cross‐entropy between π and softmax(logits)
        logp = F.log_softmax(logits, dim=1)  # [B, E]
        loss_p = - (pi_batch * logp).sum(dim=1).mean()

        # 2) Value loss: MSE between z and values
        loss_v = F.mse_loss(values, z_batch)

        loss = loss_p + loss_v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B

    return total_loss / len(dataloader.dataset)

def main_training_loop(n, device, total_iters=200, games_per_iter=25):
    # Initialize:
    net_best = GNN(n=n, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(net_best.parameters(), lr=5e-4, weight_decay=1e-5)
    buffer = ReplayBuffer(capacity=200000)

    # MCTS params (can tweak)
    mcts_params = {
        "c_puct": 1.0,
        "n_playout": 400,
        "device": device,
    }

    for iteration in range(1, total_iters + 1):
        # 1) Generate self‐play data
        data_list = generate_selfplay_data(net_best, n, games_per_iter, mcts_params)
        # 2) Push to buffer
        for (s, eidx, lmask, pi, z) in data_list:
            buffer.push(s, eidx, lmask, pi, z)

        # 3) Sample a dataset of size = min(len(buffer), 50k) for training
        sample_size = min(len(buffer), 50000)
        sample_data = random.sample(buffer.buffer, sample_size)
        dataset = DotBShotDataset(sample_data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # 4) Train a new network clone
        net_temp = GNN(n=n, hidden_dim=64).to(device)
        net_temp.load_state_dict(net_best.state_dict())  # start from best
        optimizer_temp = torch.optim.Adam(net_temp.parameters(), lr=5e-4, weight_decay=1e-5)

        # Train for a few epochs (e.g. 5 epochs)
        for epoch in range(1, 6):
            loss_val = train_network(net_temp, optimizer_temp, dataloader, device)

        # 5) Evaluate net_temp vs. net_best in 20 games
        win_temp = 0
        for game_id in range(20):
            # Alternate who goes first to avoid bias:
            first_player = +1 if game_id < 10 else -1
            winner = play_match(net_temp, net_best, n, mcts_params, first_player)
            if winner == +1:
                win_temp += 1

        # 6) If net_temp wins ≥ 12/20, accept it
        if win_temp >= 12:
            net_best = net_temp
            print(f"Iteration {iteration}: New best network accepted (won {win_temp}/20).")
        else:
            print(f"Iteration {iteration}: Keeping old best (temp won {win_temp}/20).")

        # 7) Periodic evaluation vs. baseline (random, greedy chain)
        if iteration % 5 == 0:
            win_rand = evaluate_against_random(net_best, n, mcts_params, num_games=200)
            win_greedy = evaluate_against_greedy_chain(net_best, n, mcts_params, num_games=200)
            print(f" Iter {iteration}: Win vs. random = {win_rand}/200; vs. greedy = {win_greedy}/200")

    # At end, save final model
    torch.save(net_best.state_dict(), f"gnn_dotbox_n{n}_best.pth")
# train/train_loop.py (continued)
import random

def play_match(net_A, net_B, n, mcts_params, first_player=+1):
    """
    Plays one game where net_A controls player=first_player and net_B controls the other.
    Returns +1 if net_A wins, –1 if net_B wins, 0 if tie.
    """
    env = game_env(n=n)
    players = {+1: net_A, -1: net_B}
    mcts_A = MCTS(net_A, **mcts_params)
    mcts_B = MCTS(net_B, **mcts_params)

    env.current_player = first_player
    while True:
        current_net = players[env.current_player]
        mcts_current = mcts_A if env.current_player == first_player else mcts_B
        action, _ = mcts_current.get_move(env)
        _, reward, done, info = env.step(action)
        if done:
            a_score, b_score = info["boxes"]
            if a_score > b_score:
                return +1 if first_player == +1 else -1
            elif a_score < b_score:
                return -1 if first_player == +1 else +1
            else:
                return 0
        # continue loop

def evaluate_against_random(net, n, mcts_params, num_games=200):
    wins = 0
    for i in range(num_games):
        # Random plays the other side
        env = game_env(n=n)
        current_player = +1 if (i % 2 == 0) else -1
        env.current_player = current_player
        mcts_net = MCTS(net, **mcts_params)

        while True:
            if env.current_player == current_player:
                action, _ = mcts_net.get_move(env)
            else:
                # Random opponent
                legal = env._get_observation()[1]
                action = random.choice(np.where(legal == 1)[0])

            _, reward, done, info = env.step(action)
            if done:
                a_score, b_score = info["boxes"]
                net_score = a_score if current_player == +1 else b_score
                opp_score = b_score if current_player == +1 else a_score
                if net_score > opp_score:
                    wins += 1
                break
    return wins

def evaluate_against_greedy_chain(net, n, mcts_params, num_games=200):
    """
    Greedy chain‐aware opponent:  
       - If it can close any 1‐chain (i.e. slack[i] > 0), pick that randomly.  
       - Else pick random legal move.
    """
    wins = 0
    for i in range(num_games):
        env = game_env(n=n)
        current_player = +1 if (i % 2 == 0) else -1
        env.current_player = current_player
        mcts_net = MCTS(net, **mcts_params)

        while True:
            if env.current_player == current_player:
                action, _ = mcts_net.get_move(env)
            else:
                # Greedy chain: look at f3 slack feature
                _, legal_mask = env._get_observation()
                slack = env.f3  # f3 is array of slack counts
                # candidates with slack > 0 & legal
                chain_moves = [i for i in range(env.E) if slack[i] > 0 and legal_mask[i] == 1]
                if chain_moves:
                    action = random.choice(chain_moves)
                else:
                    action = random.choice(np.where(legal_mask == 1)[0])

            _, reward, done, info = env.step(action)
            if done:
                a_score, b_score = info["boxes"]
                net_score = a_score if current_player == +1 else b_score
                opp_score = b_score if current_player == +1 else a_score
                if net_score > opp_score:
                    wins += 1
                break
    return wins
