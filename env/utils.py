# utils.py

import numpy as np


def build_line_graph(n):
    """
    Build adjacency list of the line graph for n x n board.
    Returns: adj_list: Python list of length E.
    """
    # Number of horizontal and vertical edges
    h_count = (n + 1) * n       # (n+1) rows of n horizontal edges
    v_count = n * (n + 1)       # n rows of (n+1) vertical edges
    E = h_count + v_count       # total edges = nodes in the line‐graph

    # For each dot (r,c), we will collect which edges touch that dot.
    # Dots lie in an (n+1) x (n+1) grid.
    dot_edges = [[[] for _ in range(n + 1)] for __ in range(n + 1)]

    def decode_edge(edge_idx):
        """
        Decode a 0 <= edge_idx < E into (etype, r, c), where
          etype == 'H' for horizontal, or 'V' for vertical.
        - Horizontal edges are indexed 0 .. h_count-1:
            row = edge_idx // n,   col = edge_idx % n
            (r in [0..n], c in [0..n-1])
        - Vertical edges are indexed h_count .. h_count+v_count-1:
            idx2 = edge_idx - h_count
            row = idx2 // (n+1),  col = idx2 % (n+1)
            (r in [0..n-1], c in [0..n])
        """
        if edge_idx < h_count:
            r = edge_idx // n
            c = edge_idx % n
            return ('H', r, c)
        else:
            idx2 = edge_idx - h_count
            r = idx2 // (n + 1)
            c = idx2 % (n + 1)
            return ('V', r, c)

    # 1) Populate dot_edges: for each edge_idx, figure out its two endpoint dots, and append.
    for edge_idx in range(E):
        etype, r, c = decode_edge(edge_idx)
        if etype == 'H':
            # horizontal edge at (r, c) spans dots (r, c) and (r, c+1)
            dot_edges[r][c].append(edge_idx)
            dot_edges[r][c + 1].append(edge_idx)
        else:
            # vertical edge at (r, c) spans dots (r, c) and (r+1, c)
            dot_edges[r][c].append(edge_idx)
            dot_edges[r + 1][c].append(edge_idx)

    # 2) Build adjacency: two edges are neighbors if they share a dot
    adj_list = [[] for _ in range(E)]
    for edge_idx in range(E):
        etype, r, c = decode_edge(edge_idx)
        if etype == 'H':
            dot1 = (r, c)
            dot2 = (r, c + 1)
        else:
            dot1 = (r, c)
            dot2 = (r + 1, c)

        neighbor_set = set()
        for (dr, dc) in (dot1, dot2):
            for e2 in dot_edges[dr][dc]:
                if e2 != edge_idx:
                    neighbor_set.add(e2)
        adj_list[edge_idx] = list(neighbor_set)

    return adj_list

def compute_all_slacks(f1, n):
    """
    For each edge i (0 <= i < E), look at the up to two 1×1 cells (boxes) that i borders.
    If exactly 3 of that box’s edges are drawn, that box is “three‐sided” (i.e. drawing i would complete it).
    The slack count for edge i = number of adjacent boxes that are already three‐sided.

    Parameters:
      - f1: length‐E numpy array of 0/1, where f1[i] == 1 means “edge i is drawn”
      - n: board size (n×n)

    Returns:
      - slack: length‐E numpy array of integers in {0,1,2}
        slack[i] = how many adjacent boxes of edge i already have exactly 3 drawn edges.
    """
    E = 2 * n * (n + 1)
    slack = np.zeros(E, dtype=np.int8)

    for edge_idx in range(E):
        # Look up to two adjacent cells that share this edge
        cells = adjacent_boxes_of_edge(edge_idx, n)
        cnt = 0
        for cell in cells:
            if count_edges_drawn(cell, f1, n) == 3:
                cnt += 1
        slack[edge_idx] = cnt

    return slack


def adjacent_boxes_of_edge(edge_idx, n):
    """
    Given a single edge index (0 <= edge_idx < E), return a list
    of up to two cell‐indices (in [0 .. n*n-1]) that this edge borders.  
    Each 1×1 “box” (cell) is indexed row* n + col, where row,col in [0..n-1].
    """
    h_count = (n + 1) * n
    cells = []

    if edge_idx < h_count:
        # Horizontal edge
        r = edge_idx // n
        c = edge_idx % n
        # It can border a cell above (r-1, c) if r > 0
        if r > 0:
            cells.append((r - 1) * n + c)
        # It can border a cell below (r, c) if r < n
        if r < n:
            cells.append(r * n + c)
    else:
        # Vertical edge
        idx2 = edge_idx - h_count
        r = idx2 // (n + 1)
        c = idx2 % (n + 1)
        # It can border a cell to the left (r, c-1) if c > 0
        if c > 0:
            cells.append(r * n + (c - 1))
        # It can border a cell to the right (r, c) if c < n
        if c < n:
            cells.append(r * n + c)

    return cells


def count_edges_drawn(cell_idx, f1, n):
    """
    For a given 1×1 cell index (0 <= cell_idx < n*n), count how many of its four
    surrounding edges are already drawn.  

    Parameters:
      - cell_idx: integer in [0..n*n-1], representing the 1×1 cell at row = cell_idx // n, col = cell_idx % n
      - f1:    1D numpy array of length E = 2*n*(n+1), where f1[i]==1 if edge i is drawn, else 0
      - n:     board size (n×n)

    Returns:
      - an integer in {0,1,2,3,4} indicating how many of the four edges around that cell are drawn.
    """
    # Number of horizontal edges = (n+1)*n
    h_count = (n + 1) * n

    row = cell_idx // n
    col = cell_idx % n

    total = 0
    # Top edge: horizontal at (row, col)    ⇒ edge index = row * n + col
    top_idx = row * n + col
    total += int(f1[top_idx])

    # Bottom edge: horizontal at (row+1, col) ⇒ edge index = (row+1) * n + col
    bottom_idx = (row + 1) * n + col
    total += int(f1[bottom_idx])

    # Left edge: vertical at (row, col)      ⇒ edge index = h_count + row*(n+1) + col
    left_idx = h_count + row * (n + 1) + col
    total += int(f1[left_idx])

    # Right edge: vertical at (row, col+1)    ⇒ edge index = h_count + row*(n+1) + (col+1)
    right_idx = h_count + row * (n + 1) + (col + 1)
    total += int(f1[right_idx])

    return total



def count_adjacent_three_sided_boxes(edge_idx, f1, n):
    """
    For a given edge index, look at each cell (1×1 box) it touches.
    Count how many of those cells already have exactly 3 drawn edges.
    Returns an integer in {0, 1, 2}.
    """
    count = 0
    cells = adjacent_boxes_of_edge(edge_idx, n)
    for cell in cells:
        if count_edges_drawn(cell, f1, n) == 3:
            count += 1
    return count
