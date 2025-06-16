import imageio
import numpy as np
from env.game_env import game_env
from models.gnn_net import GNN
from mcts.mcts import MCTS
import torch
from PIL import Image, ImageDraw

# Config
BOARD_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GNN(BOARD_SIZE, hidden_dim=64).to(DEVICE)
net.load_state_dict(torch.load(f"gnn_dotbox_n{BOARD_SIZE}_best.pth", map_location=DEVICE))
mcts = MCTS(net, c_puct=1.0, n_playout=200, device=DEVICE.type)

def render_frame(env, size=300):
    n = env.n
    W = size
    M = 20
    dot_r = 4
    line_w = 3
    im = Image.new("RGB", (W, W), "white")
    draw = ImageDraw.Draw(im)
    sp = (W - 2*M) / n
    # dots
    for i in range(n+1):
        for j in range(n+1):
            x, y = M + j*sp, M + i*sp
            draw.ellipse((x-dot_r,y-dot_r,x+dot_r,y+dot_r), fill="black")
    # edges
    for e in range(env.E):
        if env.f1[e]==0: continue
        etype, r, c = env.decode_edge(e)
        if etype=="H":
            x1,y1 = M+c*sp, M+r*sp
            x2,y2 = M+(c+1)*sp, y1
        else:
            x1,y1 = M+c*sp, M+r*sp
            x2,y2 = x1, M+(r+1)*sp
        col = "red" if env.f2[e]==+1 else "blue"
        draw.line((x1,y1,x2,y2), fill=col, width=line_w)
    return np.array(im)

def main():
    env = game_env(BOARD_SIZE)
    frames = []
    env.reset()
    frames.append(render_frame(env))
    # Self-play one game
    while not env.f1.all():
        action, _ = mcts.get_move(env)
        env.step(action)
        frames.append(render_frame(env))
    # Write GIF
    imageio.mimsave("agent_play.gif", frames, duration=0.5)

if __name__=="__main__":
    main()
