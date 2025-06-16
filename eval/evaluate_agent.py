# eval/evaluate_agent.py
import argparse
import random
import numpy as np
import torch
from multiprocessing import Pool
from tqdm import tqdm
from env.game_env import game_env
from models.gnn_net import GNN
from mcts.mcts import MCTS


def play_human(env):
    """
    CLI human play: prints ASCII board, prompts for edge index.
    Returns final scores when done.
    """
    while True:
        n = env.n
        grid = []
        idx_map = {}
        for r in range(2*n+1):
            row = []
            for c in range(2*n+1):
                if r%2==0 and c%2==0:
                    row.append('.')
                elif r%2==0 and c%2==1:
                    e = (r//2)*n + (c//2)
                    row.append('-' if env.f1[e] else str(e%10))
                    idx_map[(r,c)] = e
                elif r%2==1 and c%2==0:
                    e = (n+1)*n + (r//2)*(n+1) + (c//2)
                    row.append('|' if env.f1[e] else str(e%10))
                    idx_map[(r,c)] = e
                else:
                    row.append(' ')
            grid.append(row)
        for row in grid:
            print(' '.join(row))
        if env.f1.sum() == env.E:
            break
        try:
            choice = int(input("Enter edge index: "))
        except ValueError:
            print("Invalid input.")
            continue
        if choice<0 or choice>=env.E or env.f1[choice]:
            print("Illegal move.")
            continue
        env.step(choice)
    return env.box_counts[+1], env.box_counts[-1]


def eval_game(seed, board_size, playouts, device_str, checkpoint, opp):
    # initialize environment and agent for a single game
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = game_env(n=board_size)
    env.reset()
    env.current_player = +1 if seed % 2 == 0 else -1

    device = torch.device(device_str)
    net = GNN(n=board_size, hidden_dim=32).to(device)
    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state)
    net.eval()

    mcts = MCTS(net, c_puct=1.0, n_playout=playouts, device=device_str)
    wins = 0
    while True:
        if env.current_player == +1:
            action, _ = mcts.get_move(env)
        else:
            legal = env._get_observation()[1]
            if opp == 'random':
                action = random.choice(np.where(legal == 1)[0])
            else:
                slack = env.f3
                chain = [e for e in range(env.E) if slack[e] > 0 and legal[e] == 1]
                action = random.choice(chain) if chain else random.choice(np.where(legal == 1)[0])
        _, _, done, info = env.step(action)
        if done:
            _boxes = info.get("boxes")
            if _boxes is None:
                raise KeyError(f"Expected key 'boxes' in info, got {info!r}")
            if not (isinstance(_boxes, tuple) and len(_boxes) == 2):
                raise ValueError(f"Expected info['boxes'] to be 2-tuple, got: {_boxes!r}")
            a, b = _boxes
            wins = 1 if a > b else 0
            break
    return wins


def eval_game_star(params):
    # helper for multiprocessing: unpack and call eval_game
    return eval_game(*params)


def main():
    parser = argparse.ArgumentParser("Evaluate Dots & Boxes agent")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--board_size", type=int, default=5)
    parser.add_argument("--device", choices=['cpu','cuda'], default='cpu')
    parser.add_argument("--opp", choices=['random','greedy_chain','human'], required=True)
    parser.add_argument("--num_games", type=int, default=200)
    parser.add_argument("--playouts", type=int, default=400)
    args = parser.parse_args()

    if args.opp == 'human':
        env = game_env(n=args.board_size)
        env.reset()
        print("Human plays +1, agent plays -1")
        a, b = play_human(env)
        print(f"Final: You={a}, Agent={b}")
        return

    print(f"Evaluating {args.num_games} games vs {args.opp} with {args.playouts} playouts on {args.device}")
    params = [
        (i, args.board_size, args.playouts, args.device, args.checkpoint, args.opp)
        for i in range(args.num_games)
    ]
    results = []
    # On GPU, avoid multiprocessing fork issues: run sequentially
    if args.device == 'cuda':
        for p in tqdm(params, total=args.num_games, desc="Games"):
            results.append(eval_game_star(p))
    else:
        with Pool() as pool:
            for win in tqdm(
                pool.imap_unordered(eval_game_star, params),
                total=args.num_games,
                desc="Games"
            ):
                results.append(win)
    wins = sum(results)
    print(f"Agent wins: {wins}/{args.num_games}")

if __name__ == '__main__':
    main()
