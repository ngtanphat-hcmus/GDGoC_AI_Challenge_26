import time
import random
import numpy as np
from collections import deque

class SimState:
    def __init__(self, grid, px, py, alive, bombs_left, max_bombs, bomb_radius, bombs, stats, enemies):
        self.grid = grid.copy()
        self.px = px
        self.py = py
        self.alive = alive
        self.bombs_left = bombs_left
        self.max_bombs = max_bombs
        self.bomb_radius = bomb_radius
        self.bombs = [list(b) for b in bombs] # List of [x, y, timer, owner_id, radius]
        self.stats = stats.copy()
        self.enemies = list(enemies) # List of (ex, ey, alive)

    def get_blast_tiles(self, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x = bx + dx * r
                y = by + dy * r
                if not (0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]):
                    break
                cell = self.grid[x, y]
                if cell == 1: # WALL
                    break
                tiles.add((x, y))
                if cell == 2: # BOX
                    break
        return tiles

    def step(self, action):
        if not self.alive:
            return

        dx, dy = 0, 0
        if action == 1: dx = -1  # LEFT -> Visually UP (row decreases)
        elif action == 2: dx = 1  # RIGHT -> Visually DOWN (row increases)
        elif action == 3: dy = -1 # UP -> Visually LEFT (col decreases)
        elif action == 4: dy = 1  # DOWN -> Visually RIGHT (col increases)
        elif action == 5:
            # Place bomb
            if self.bombs_left > 0:
                has_bomb = any(b[0] == self.px and b[1] == self.py for b in self.bombs)
                if not has_bomb:
                    self.bombs.append([self.px, self.py, 7, 0, self.bomb_radius])
                    self.bombs_left -= 1
                    self.stats['bombs'] += 1

        if dx != 0 or dy != 0:
            nx = self.px + dx
            ny = self.py + dy
            if 0 < nx < self.grid.shape[0] - 1 and 0 < ny < self.grid.shape[1] - 1:
                if self.grid[nx, ny] not in [1, 2]: # not wall or box
                    blocked_by_bomb = any(b[0] == nx and b[1] == ny for b in self.bombs)
                    if not blocked_by_bomb:
                        self.px = nx
                        self.py = ny
                        cell = self.grid[nx, ny]
                        if cell == 3: # ITEM_RADIUS
                            self.bomb_radius = min(self.bomb_radius + 1, 5)
                            self.grid[nx, ny] = 0
                            self.stats['items'] += 1
                        elif cell == 4: # ITEM_CAPACITY
                            self.max_bombs = min(self.max_bombs + 1, 5)
                            self.bombs_left = min(self.bombs_left + 1, self.max_bombs)
                            self.grid[nx, ny] = 0
                            self.stats['items'] += 1

        # Tick bombs
        exploded_this_step = []
        for b in self.bombs:
            b[2] -= 1
            if b[2] <= 0:
                exploded_this_step.append(b)

        # Chain reaction
        idx = 0
        while idx < len(exploded_this_step):
            b = exploded_this_step[idx]
            idx += 1
            blast = self.get_blast_tiles(b[0], b[1], b[4])
            for other in self.bombs:
                if other in exploded_this_step:
                    continue
                if (other[0], other[1]) in blast:
                    other[2] = 0
                    exploded_this_step.append(other)

        # Apply explosions
        if exploded_this_step:
            affected = set()
            for b in exploded_this_step:
                blast = self.get_blast_tiles(b[0], b[1], b[4])
                affected.update(blast)
                if b[3] == 0:
                    self.bombs_left = min(self.bombs_left + 1, self.max_bombs)

            if (self.px, self.py) in affected:
                self.alive = False

            # Check enemies
            for i, (ex, ey, e_alive) in enumerate(self.enemies):
                if e_alive and (ex, ey) in affected:
                    self.enemies[i] = (ex, ey, False)
                    self.stats['kills'] += 1

            for tx, ty in affected:
                if self.grid[tx, ty] == 2:
                    self.grid[tx, ty] = 0
                    self.stats['boxes'] += 1
                elif self.grid[tx, ty] in [3, 4]:
                    self.grid[tx, ty] = 0

            self.bombs = [b for b in self.bombs if b[2] > 0]

def get_effective_bombs(bombs, grid):
    eff_bombs = [list(b) for b in bombs]
    changed = True
    while changed:
        changed = False
        for i in range(len(eff_bombs)):
            b1 = eff_bombs[i]
            bx, by, timer1, _, radius1 = b1
            blast = {(bx, by)}
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for r in range(1, radius1 + 1):
                    x = bx + dx * r
                    y = by + dy * r
                    if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                        break
                    cell = grid[x, y]
                    if cell == 1:
                        break
                    blast.add((x, y))
                    if cell == 2:
                        break
            for j in range(len(eff_bombs)):
                if i == j:
                    continue
                b2 = eff_bombs[j]
                if (b2[0], b2[1]) in blast:
                    if b2[2] > timer1:
                        b2[2] = timer1
                        changed = True
    return eff_bombs

def get_pos_danger_timer(pos, bombs, grid):
    min_timer = 99
    for b in bombs:
        bx, by, timer, _, radius = b
        blast = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x = bx + dx * r
                y = by + dy * r
                if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                    break
                cell = grid[x, y]
                if cell == 1:
                    break
                blast.add((x, y))
                if cell == 2:
                    break
        if pos in blast:
            if timer < min_timer:
                min_timer = timer
    return min_timer

def has_escape_path(state):
    if not state.bombs:
        return True
    blast_tiles = set()
    for b in state.bombs:
        bx, by, _, _, radius = b
        blast_tiles.update(state.get_blast_tiles(bx, by, radius))

    if (state.px, state.py) not in blast_tiles:
        return True

    q = deque([(state.px, state.py)])
    seen = {(state.px, state.py)}
    while q:
        pos = q.popleft()
        if pos not in blast_tiles:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < state.grid.shape[0] and 0 <= ny < state.grid.shape[1]:
                if state.grid[nx, ny] in [0, 3, 4]:
                    if not any(b[0] == nx and b[1] == ny for b in state.bombs):
                        npos = (nx, ny)
                        if npos not in seen:
                            seen.add(npos)
                            q.append(npos)
    return False

def bfs_dist(grid, start, target_values):
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if grid[pos[0], pos[1]] in target_values:
            return d
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] in [0, 3, 4]:
                    npos = (nx, ny)
                    if npos not in seen:
                        seen.add(npos)
                        q.append((npos, d + 1))
    return None

def bfs_dist_to_positions(grid, start, target_positions):
    if not target_positions:
        return None
    target_set = set(target_positions)
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if pos in target_set:
            return d
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] in [0, 3, 4]:
                    npos = (nx, ny)
                    if npos not in seen:
                        seen.add(npos)
                        q.append((npos, d + 1))
    return None

def get_box_neighbors(grid):
    spots = set()
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 2:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if grid[nx, ny] in [0, 3, 4]:
                            spots.add((nx, ny))
    return spots

class Agent:
    team_id = "AlphaSearchAgent"
    
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

    def evaluate_state(self, state):
        if not state.alive:
            return -100000.0

        if not has_escape_path(state):
            return -20000.0

        score = 0.0
        score += 10000.0 # Alive bonus

        # Stats progress
        score += state.stats['kills'] * 8000.0
        score += state.stats['boxes'] * 200.0
        score += state.stats['items'] * 400.0

        # Parameters bonus
        score += state.bomb_radius * 80.0
        score += state.max_bombs * 80.0

        # Heuristic distances
        px, py = state.px, state.py
        grid = state.grid

        item_dist = bfs_dist(grid, (px, py), [3, 4])
        box_spots = get_box_neighbors(grid)
        box_dist = bfs_dist_to_positions(grid, (px, py), box_spots)

        enemy_positions = [(ex, ey) for ex, ey, e_alive in state.enemies if e_alive]
        enemy_dist = bfs_dist_to_positions(grid, (px, py), enemy_positions)

        # Late-game behavior adjustments
        if self.step_count > 400:
            # Tie-break priority: Kills > Boxes > Items > Bombs
            if enemy_dist is not None:
                score += (15 - enemy_dist) * 40.0
            if box_dist is not None:
                score += (15 - box_dist) * 30.0
            if item_dist is not None:
                score += (15 - item_dist) * 20.0
            score += state.stats['bombs'] * 10.0
        else:
            # Mid-game prioritizes items and farming
            if item_dist is not None:
                score += (15 - item_dist) * 30.0
            if box_dist is not None:
                score += (15 - box_dist) * 20.0
            if enemy_dist is not None:
                score += (15 - enemy_dist) * 10.0

        # Penalty for standing in danger
        eff_bombs = get_effective_bombs(state.bombs, state.grid)
        danger_timer = get_pos_danger_timer((px, py), eff_bombs, state.grid)
        if danger_timer <= 2:
            score -= (3 - danger_timer) * 5000.0

        return score

    def search(self, state, depth, start_time, time_limit):
        if depth == 0 or not state.alive:
            return self.evaluate_state(state), None

        if time.time() - start_time > time_limit:
            return self.evaluate_state(state), None

        best_score = -99999999.0
        best_action = 0

        # Prioritize moves to improve ordering and cutoffs
        actions = [0, 1, 2, 3, 4, 5]
        
        # Simple action validity filters
        valid_actions = []
        for action in actions:
            if action == 5:
                if state.bombs_left <= 0:
                    continue
                if any(b[0] == state.px and b[1] == state.py for b in state.bombs):
                    continue
            elif action in [1, 2, 3, 4]:
                dx, dy = 0, 0
                if action == 1: dx = -1
                elif action == 2: dx = 1
                elif action == 3: dy = -1
                elif action == 4: dy = 1
                nx = state.px + dx
                ny = state.py + dy
                if not (0 < nx < state.grid.shape[0] - 1 and 0 < ny < state.grid.shape[1] - 1):
                    continue
                if state.grid[nx, ny] in [1, 2]:
                    continue
                if any(b[0] == nx and b[1] == ny for b in state.bombs):
                    continue
            valid_actions.append(action)

        if not valid_actions:
            return -100000.0, 0

        for action in valid_actions:
            # Simulate the transition
            next_state = SimState(
                state.grid, state.px, state.py, state.alive,
                state.bombs_left, state.max_bombs, state.bomb_radius,
                state.bombs, state.stats, state.enemies
            )
            next_state.step(action)

            score, _ = self.search(next_state, depth - 1, start_time, time_limit)
            if score > best_score:
                best_score = score
                best_action = action

        return best_score, best_action

    def act(self, obs):
        start_time = time.time()
        self.step_count += 1

        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        max_bombs = 1 + int(players[self.agent_id][3]) - int(bombs_left) # max capacity calculation
        # Wait, the obs contains: [row, col, alive, bombs_left, bomb_radius_bonus]
        # Let's verify how max capacity is retrieved.
        # Actually we don't have max capacity directly in obs, but we can track it!
        # When we start max_bombs = 1. Every time we collect CAPACITY item (4), max_bombs increases.
        # We can also compute max_bombs = bombs_left + count_owned_active_bombs
        owned_active_bombs = sum(1 for b in bombs if int(b[3]) == self.agent_id)
        max_bombs = int(bombs_left) + owned_active_bombs

        # Prepare bombs list in [x, y, timer, owner_id, radius] format
        parsed_bombs = []
        for b in bombs:
            bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            # get radius of the bomb based on owner stats
            b_radius = 2 # default
            if 0 <= owner_id < len(players):
                b_radius = max(1, int(players[owner_id][4]) + 1)
            parsed_bombs.append([bx, by, timer, owner_id, b_radius])

        enemies = []
        for i, p in enumerate(players):
            if i != self.agent_id:
                enemies.append((int(p[0]), int(p[1]), bool(p[2] == 1)))

        stats = {'kills': 0, 'boxes': 0, 'items': 0, 'bombs': 0}

        initial_state = SimState(
            grid, my_pos[0], my_pos[1], True,
            int(bombs_left), max_bombs, bomb_radius,
            parsed_bombs, stats, enemies
        )

        # Run Iterative Deepening search within 80ms
        best_action = 0
        time_limit = 0.080
        for depth in range(1, 6):
            if time.time() - start_time > time_limit:
                break
            score, action = self.search(initial_state, depth, start_time, time_limit)
            if action is not None:
                best_action = action

        return best_action
