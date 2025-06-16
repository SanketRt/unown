import pygame, sys
import torch
from env.game_env import game_env
from mcts.mcts import MCTS
from models.gnn_net import GNN

# ─── Config ─────────────────────────────────────────────────────────────
BOARD_SIZE = 5
SCREEN_SIZE = 600                   
MARGIN = 50
DOT_RADIUS = 5
LINE_WIDTH = 4
AGENT_PLAYOUTS = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────

def draw_board(screen, env):
    screen.fill((255,255,255))
    n = env.n
    spacing = (SCREEN_SIZE - 2*MARGIN) / n
    # Draw dots
    for i in range(n+1):
        for j in range(n+1):
            x = MARGIN + j*spacing
            y = MARGIN + i*spacing
            pygame.draw.circle(screen, (0,0,0), (int(x),int(y)), DOT_RADIUS)
    # Draw edges
    for e in range(env.E):
        if env.f1[e] == 0: continue
        etype, r, c = env.decode_edge(e)
        if etype == 'H':
            x1 = MARGIN + c*spacing
            y1 = MARGIN + r*spacing
            x2 = MARGIN + (c+1)*spacing
            y2 = y1
        else:
            x1 = MARGIN + c*spacing
            y1 = MARGIN + r*spacing
            x2 = x1
            y2 = MARGIN + (r+1)*spacing
        color = (200,0,0) if env.f2[e] == +1 else (0,0,200)
        pygame.draw.line(screen, color, (x1,y1), (x2,y2), LINE_WIDTH)
    pygame.display.flip()

def edge_from_mouse(pos, env):
    # find the closest *legal* edge to click position
    mx, my = pos
    n = env.n
    spacing = (SCREEN_SIZE - 2*MARGIN) / n
    best, best_dist = None, float('inf')
    for e in range(env.E):
        if env.f1[e] == 1: continue
        etype, r, c = env.decode_edge(e)
        if etype == 'H':
            x1 = MARGIN + c*spacing; y1 = MARGIN + r*spacing
            x2 = MARGIN + (c+1)*spacing; y2 = y1
        else:
            x1 = MARGIN + c*spacing; y1 = MARGIN + r*spacing
            x2 = x1; y2 = MARGIN + (r+1)*spacing
        # distance point-to-segment
        px, py = mx - x1, my - y1
        vx, vy = x2 - x1, y2 - y1
        t = max(0, min(1, (px*vx+py*vy) / (vx*vx+vy*vy)))
        cx, cy = x1 + t*vx, y1 + t*vy
        dist = ((mx-cx)**2 + (my-cy)**2)**0.5
        if dist < best_dist:
            best, best_dist = e, dist
    # threshold click radius
    return best if best_dist < LINE_WIDTH*2 else None

def main():
    # Load agent
    env = game_env(BOARD_SIZE)
    net = GNN(n=BOARD_SIZE, hidden_dim=64).to(DEVICE)
    net.load_state_dict(torch.load(f"gnn_dotbox_n{BOARD_SIZE}_best.pth", map_location=DEVICE))
    mcts = MCTS(net, c_puct=1.0, n_playout=AGENT_PLAYOUTS, device=DEVICE.type)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Dots & Boxes vs. GNN-MCTS")

    env.reset()
    human_player = +1  # human plays +1, agent plays -1

    draw_board(screen, env)
    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and env.current_player == human_player:
                e = edge_from_mouse(event.pos, env)
                if e is not None:
                    _, _, done, _ = env.step(e)
                    draw_board(screen, env)
        # Agent’s turn
        if env.current_player != human_player and not env.f1.all():
            action, _ = mcts.get_move(env)
            env.step(action)
            draw_board(screen, env)
        # Check terminal
        if env.f1.all():
            a_score, b_score = env.box_counts[+1], env.box_counts[-1]
            text = f"You: {a_score}  Agent: {b_score}"
            img = font.render(text, True, (0,0,0))
            screen.blit(img, (MARGIN, SCREEN_SIZE-30))
            pygame.display.flip()
            pygame.time.wait(5000)
            running = False

    pygame.quit()
    sys.exit()

if __name__=="__main__":
    main()
