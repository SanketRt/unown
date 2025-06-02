````markdown
# Dots & Boxes GNN + MCTS

A Graph Neural Network (GNN) with Monte Carlo Tree Search (MCTS) agent for n×n Dots & Boxes.

---

## 📦 Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/yourusername/dots_and_boxes_gnn.git
   cd dots_and_boxes_gnn
````

2. **Create & activate a virtualenv**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch & PyG**

   * GPU (CUDA 11.7 example):

     ```bash
     pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
     pip install torch_geometric
     ```
   * CPU-only:

     ```bash
     pip install torch torchvision
     pip install torch_geometric
     ```

4. **Install other dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Folder Layout

```
dots_and_boxes_gnn/
├─ env/
│  ├ game_env.py      # Environment (line-graph, step())
│  └ utils.py         # build_line_graph, slack, etc.
├─ models/
│  └ gnn_net.py       # GNN definition (GCNConv → policy/value)
├─ mcts/
│  ├ node.py          # MCTSNode (N, W, P, Q)
│  └ mcts.py          # Selection, expansion, backprop
├─ train/
│  ├ replay_buffer.py # FIFO buffer for (state, π, z)
│  ├ selfplay.py      # run_selfplay_one_game()
│  └ train_loop.py    # main loop: self-play, train, evaluate
├─ eval/
│  └ evaluate_agent.py# Play vs. random/greedy/human
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Quick Start

### 1. Train

Edit hyperparameters in `scripts/run_train.sh` (e.g. `BOARD_SIZE=5`, `TOTAL_ITERS=100`). Then:

```bash
bash scripts/run_train.sh
```

This runs:

```bash
python train/train_loop.py \
  --board_size 5 \
  --total_iters 100 \
  --games_per_iter 25 \
  --device cuda
```

Checkpoints are saved as `gnn_dotbox_n${BOARD_SIZE}_best.pth`.

---

### 2. Evaluate

**Vs. random**:

```bash
python eval/evaluate_agent.py \
  --checkpoint gnn_dotbox_n5_best.pth \
  --board_size 5 \
  --opp random \
  --num_games 200
```

**Vs. greedy‐chain**:

```bash
python eval/evaluate_agent.py \
  --checkpoint gnn_dotbox_n5_best.pth \
  --board_size 5 \
  --opp greedy_chain \
  --num_games 200
```

**Play vs. human (CLI)**:

```bash
python eval/evaluate_agent.py \
  --checkpoint gnn_dotbox_n5_best.pth \
  --board_size 5 \
  --opp human
```

---

## 🧠 Hyperparameters (Defaults)

* **GNN**

  * Hidden dim = 64, 2 GCNConv layers
  * Adam lr = 5 × 10⁻⁴, weight\_decay = 1 × 10⁻⁵
  * Batch size = 256

* **MCTS**

  * c\_puct = 1.0
  * n\_playout = 400 (5×5), 800 (7×7)

* **Self-Play**

  * games\_per\_iter = 25
  * buffer\_capacity = 200 000
  * sample\_size = 50 000
  * train\_epochs = 5
  * eval\_match\_size = 20 (accept if temp wins ≥ 12)

* **Evaluation**

  * vs. random/greedy: 200 games

Adjust for your compute/GPU.

---

## 🚀 Overview

1. **Environment** (`env/game_env.py`):

   * Builds line-graph: E = 2 · n · (n+1) nodes.
   * State = `[f1, f2, f3]` arrays (drawn, ownership, slack).
   * `step(action)` returns `(obs, reward, done, info)` with `info["boxes"]` at terminal.

2. **GNN** (`models/gnn_net.py`):

   ```python
   class DotBoxGNN(nn.Module):
       def __init__(self, n, hidden_dim=64):
           super().__init__()
           self.E = 2 * n * (n + 1)
           self.conv1 = GCNConv(3, hidden_dim)
           self.conv2 = GCNConv(hidden_dim, hidden_dim)
           self.policy_head = nn.Linear(hidden_dim, 1)
           self.value_head  = nn.Linear(hidden_dim, 1)

       def forward(self, x, edge_index, batch=None, legal_mask=None):
           h = F.relu(self.conv1(x, edge_index))
           h = F.relu(self.conv2(h, edge_index))
           logits = self.policy_head(h).squeeze(-1)
           if legal_mask is not None:
               logits = logits.masked_fill(legal_mask == 0, float('-1e9'))
           m = global_mean_pool(h, batch) if batch is not None else h.mean(dim=0, keepdim=True)
           v = torch.tanh(self.value_head(m)).squeeze(-1)
           return logits, v
   ```

3. **MCTS** (`mcts/mcts.py` + `node.py`):

   * `MCTSNode` stores `{N, W, Q, P}` per action, and `player`.
   * `MCTS`:

     * `_evaluate(env)`: gets `(logits, v)` from GNN.
     * `_playout(root_env, root_node)`: selection → expansion (call GNN) → backprop (flip v when turn switches).
     * `get_move(env)`: run n\_playout playouts, return `(best_action, π)` where π\[a] ∝ N\[a].

4. **Training** (`train/train_loop.py`):

   * Self-play → collect `(state, edge_index, legal_mask, π, z)`.
   * Store in `ReplayBuffer`.
   * Sample minibatch, train `net_temp` for 5 epochs on policy/value loss.
   * Evaluate `net_temp` vs. `net_best` in 20 games; accept if wins ≥ 12.
   * Save best model.

---

## 🔗 References

* Joossens et al., **“Multi-Agent Learning in Canonical Games and Dots-and-Boxes (OpenSpiel)”**
* Pandey, **“Solving Dots & Boxes Using Reinforcement Learning”** (CSU 2022)
* Deakos et al., **“AlphaZero for Dots & Boxes”** (2019)
* Peters, **“Scalable AlphaZero for Dots & Boxes”** (2018)

---

Happy Training! 🎯

```
```
