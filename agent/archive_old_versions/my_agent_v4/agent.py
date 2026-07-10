"""
AlphaSearch Agent V4 — Competition-Level Bomberland AI
GDGoC-HCMUS AI Challenge 2026

Key improvements over V2/V3:
  1. Temporal escape BFS: state = (position, time), includes WAIT action
  2. Bomb memory: tracks radius at placement time (not current owner radius)
  3. Value-based decisions: every action scored numerically
  4. Anti-camping / tiebreak awareness with stat tracking
  5. Space awareness: prefers positions with more escape routes
  6. Multi-bomb chain awareness with conservative safety buffer
  7. Enemy position blocking in escape verification
  8. Robust movement bounds matching engine exactly
"""
from collections import deque

# ─── Constants ─────────────────────────────────────────────
GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4
PASSABLE_SET = {GRASS, ITEM_R, ITEM_C}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Actions: 0=STOP, 1=row-1, 2=row+1, 3=col-1, 4=col+1, 5=BOMB
MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
MOVE_ACTIONS = (1, 2, 3, 4)
ALL_ACTIONS = (1, 2, 3, 4, 0)  # movement + wait for escape BFS
INF_TIMER = 999


class Agent:
    team_id = "AlphaSearchV4"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0
        self._bomb_radii = {}       # (bx, by, owner_id) -> radius at placement
        self._last_sig = None
        # Stat tracking for tiebreak awareness
        self._kills = 0
        self._boxes = 0
        self._items = 0
        self._bombs_placed = 0

    def act(self, obs):
        try:
            return self._decide(obs)
        except Exception:
            return 0

    # ═══════════════════════════════════════════════════════
    # CORE DECISION
    # ═══════════════════════════════════════════════════════

    def _decide(self, obs):
        grid = obs["map"]
        players = obs["players"]
        raw_bombs = obs["bombs"]
        H, W = grid.shape

        # Detect new game
        if self._is_new_game(obs, H, W):
            self.step = 0
            self._bomb_radii = {}
            self._kills = 0
            self._boxes = 0
            self._items = 0
            self._bombs_placed = 0
        self.step += 1

        me = players[self.agent_id]
        if int(me[2]) != 1:
            return 0

        mx, my = int(me[0]), int(me[1])
        my_pos = (mx, my)
        bombs_left = int(me[3])
        bonus = int(me[4])
        my_radius = max(1, bonus + 1)

        # Sync bomb memory (track radius at placement time)
        self._sync_bomb_memory(raw_bombs, players)

        # Parse bombs with correct radii
        bomb_positions = set()
        bomb_list = []  # (bx, by, timer, owner_id, radius)
        for b in raw_bombs:
            bx, by, tm, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rad = self._get_bomb_radius(bx, by, oid, players)
            bomb_list.append((bx, by, tm, oid, rad))
            bomb_positions.add((bx, by))

        # Parse enemies
        enemies = []  # list of (x, y, id)
        enemy_positions = set()
        for i, p in enumerate(players):
            if i != self.agent_id and int(p[2]) == 1:
                ex, ey = int(p[0]), int(p[1])
                enemies.append((ex, ey, i))
                enemy_positions.add((ex, ey))

        # Build hazard map: tile -> minimum time until explosion
        hazard = self._build_hazard(grid, bomb_list, H, W)

        # ═══ 1 — ESCAPE (temporal BFS with WAIT) ══════════
        if my_pos in hazard:
            escape = self._escape_action(grid, my_pos, hazard, bomb_positions,
                                         enemy_positions, H, W)
            if escape is not None:
                return int(escape)
            # Fallback: move to highest-timer neighbor
            return self._desperation_move(grid, my_pos, hazard, bomb_positions,
                                          H, W)

        # ═══ 2 — EVALUATE BOMBING ═════════════════════════
        bomb_val = -INF_TIMER
        if bombs_left > 0 and my_pos not in bomb_positions and my_pos not in hazard:
            bomb_val = self._evaluate_bomb(grid, my_pos, my_radius, bomb_list,
                                           bomb_positions, enemies, enemy_positions,
                                           hazard, players, H, W)

        # ═══ 3 — SCORE ALL ACTIONS ════════════════════════
        best_action = 0
        best_score = -INF_TIMER

        # Score bombing
        if bomb_val > self._bomb_threshold():
            best_action = 5
            best_score = bomb_val

        # Score movement toward targets
        move_targets = self._find_move_targets(grid, my_pos, my_radius,
                                               bomb_positions, enemies,
                                               enemy_positions, hazard,
                                               bombs_left, bonus, H, W)

        if move_targets:
            move_action = self._route_to_scored_targets(
                grid, my_pos, move_targets, hazard, bomb_positions, H, W)
            if move_action is not None:
                # Get the score of the best target we'd reach
                move_score = max(move_targets.values())
                if move_score > best_score:
                    best_score = move_score
                    best_action = move_action

        # If nothing good, use safe idle
        if best_score <= -INF_TIMER:
            best_action = self._safe_idle_action(grid, my_pos, hazard,
                                                  bomb_positions, enemies,
                                                  H, W)

        try:
            self._last_sig = self._state_sig(obs)
        except Exception:
            self._last_sig = None

        return int(best_action)

    # ═══════════════════════════════════════════════════════
    # GAME STATE DETECTION
    # ═══════════════════════════════════════════════════════

    def _is_new_game(self, obs, H, W):
        try:
            players = obs["players"]
            bombs = obs["bombs"]
            if len(bombs) != 0 or len(players) < 4:
                return False
            expected = ((1, 1), (H - 2, W - 2), (1, W - 2), (H - 2, 1))
            for idx, pos in enumerate(expected):
                if int(players[idx][2]) != 1:
                    return False
                if (int(players[idx][0]), int(players[idx][1])) != pos:
                    return False
            sig = self._state_sig(obs)
            return sig != self._last_sig
        except Exception:
            return False

    def _state_sig(self, obs):
        players = obs.get("players", [])
        bombs = obs.get("bombs", [])
        return (tuple((int(p[0]), int(p[1]), int(p[2])) for p in players),
                len(bombs))

    # ═══════════════════════════════════════════════════════
    # BOMB MEMORY (track radius at placement time)
    # ═══════════════════════════════════════════════════════

    def _sync_bomb_memory(self, raw_bombs, players):
        active = set()
        for b in raw_bombs:
            bx, by, _, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            key = (bx, by, oid)
            active.add(key)
            if key not in self._bomb_radii:
                # First time seeing this bomb — record current owner radius
                if 0 <= oid < len(players):
                    self._bomb_radii[key] = max(1, int(players[oid][4]) + 1)
                else:
                    self._bomb_radii[key] = 1
        # Clean up exploded bombs
        for key in list(self._bomb_radii):
            if key not in active:
                del self._bomb_radii[key]

    def _get_bomb_radius(self, bx, by, oid, players):
        key = (bx, by, oid)
        r = self._bomb_radii.get(key)
        if r is not None:
            return r
        if 0 <= oid < len(players):
            return max(1, int(players[oid][4]) + 1)
        return 1

    # ═══════════════════════════════════════════════════════
    # SPATIAL HELPERS
    # ═══════════════════════════════════════════════════════

    def _in_bounds(self, x, y, H, W):
        return 0 < x < H - 1 and 0 < y < W - 1

    def _passable(self, grid, x, y, H, W):
        return self._in_bounds(x, y, H, W) and int(grid[x, y]) in PASSABLE_SET

    def _can_enter(self, grid, pos, bomb_positions, current_pos, H, W):
        """Check if we can move to pos. Can't enter bomb tiles (unless it's our current tile)."""
        if not self._passable(grid, pos[0], pos[1], H, W):
            return False
        if pos in bomb_positions and pos != current_pos:
            return False
        return True

    def _blast_tiles(self, grid, bx, by, radius, H, W):
        """Get all tiles hit by a bomb at (bx, by) with given radius."""
        tiles = [(bx, by)]
        for dx, dy in DIRS:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if x < 0 or x >= H or y < 0 or y >= W:
                    break
                cell = int(grid[x, y])
                if cell == WALL:
                    break
                tiles.append((x, y))
                if cell == BOX:
                    break
        return tiles

    def _line_clear(self, grid, a, b):
        """True if no wall/box between positions a and b (exclusive of endpoints)."""
        ax, ay = a
        bx, by = b
        if ax == bx:
            s = 1 if by > ay else -1
            for y in range(ay + s, by, s):
                if int(grid[ax, y]) in (WALL, BOX):
                    return False
            return True
        if ay == by:
            s = 1 if bx > ax else -1
            for x in range(ax + s, bx, s):
                if int(grid[x, ay]) in (WALL, BOX):
                    return False
            return True
        return False

    def _local_space_score(self, grid, pos, bomb_positions, hazard, H, W):
        """Score a position by its local openness and safety."""
        score = 0
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self._passable(grid, nx, ny, H, W) and (nx, ny) not in bomb_positions:
                score += 14
        # Bonus for items
        cell = int(grid[pos[0], pos[1]])
        if cell == ITEM_C:
            score += 70
        elif cell == ITEM_R:
            score += 60
        # Penalty for being in hazard zone
        if pos in hazard and hazard[pos] <= 3:
            score -= 90
        return score

    def _count_open_exits(self, grid, x, y, bomb_positions, H, W):
        """Count passable neighbors not blocked by bombs."""
        count = 0
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if self._passable(grid, nx, ny, H, W) and (nx, ny) not in bomb_positions:
                count += 1
        return count

    # ═══════════════════════════════════════════════════════
    # HAZARD MAP (chain-reaction aware)
    # ═══════════════════════════════════════════════════════

    def _build_hazard(self, grid, bomb_list, H, W, extra_bombs=()):
        """Build hazard map: {(x,y): min_timer_until_explosion}.
        Handles chain reactions between bombs."""
        combined = []
        for bx, by, tm, oid, rad in bomb_list:
            if tm > 0:
                combined.append([bx, by, tm, rad])
        for bx, by, tm, _oid, rad in extra_bombs:
            if tm > 0:
                combined.append([int(bx), int(by), int(tm), int(rad)])

        if not combined:
            return {}

        # Precompute blast zones
        blast_cache = []
        for entry in combined:
            blast_cache.append(set(
                self._blast_tiles(grid, entry[0], entry[1], entry[3], H, W)
            ))

        # Chain reaction propagation (fixed-point)
        n = len(combined)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                pos_i = (combined[i][0], combined[i][1])
                for j in range(n):
                    if i != j and pos_i in blast_cache[j] and combined[i][2] > combined[j][2]:
                        combined[i][2] = combined[j][2]
                        changed = True

        # Build tile -> min timer
        hazard = {}
        for i, entry in enumerate(combined):
            timer = max(1, entry[2])
            for tile in blast_cache[i]:
                if timer < hazard.get(tile, INF_TIMER):
                    hazard[tile] = timer
        return hazard

    # ═══════════════════════════════════════════════════════
    # TEMPORAL ESCAPE BFS — state = (position, time_step)
    # Includes WAIT (action 0) as valid move
    # ═══════════════════════════════════════════════════════

    def _escape_action(self, grid, my_pos, hazard, bomb_positions,
                       enemy_positions, H, W, max_depth=14,
                       require_no_hazard=True, block_enemies=True):
        """Temporal BFS escape. Returns first action to reach safety."""
        # Block enemies as obstacles (they can block our path)
        blocked = bomb_positions | enemy_positions if block_enemies else set(bomb_positions)

        queue = deque()
        queue.append((my_pos, 0, -1))  # (pos, depth, first_action)
        seen = {(my_pos, 0)}

        while queue:
            pos, depth, first_action = queue.popleft()

            # Check if this position at this time is safe destination
            if depth > 0:
                if require_no_hazard:
                    is_safe = pos not in hazard
                else:
                    is_safe = hazard.get(pos, INF_TIMER) > depth + 2
                if is_safe:
                    return first_action

            if depth >= max_depth:
                continue

            for action in ALL_ACTIONS:  # 1,2,3,4,0 (moves + wait)
                dx, dy = MOVES[action]
                npos = (pos[0] + dx, pos[1] + dy)

                if action != 0:
                    # Movement action: check can enter
                    if not self._passable(grid, npos[0], npos[1], H, W):
                        continue
                    # Can't enter bomb tiles (unless it's where we already are)
                    if npos in bomb_positions and npos != pos:
                        continue
                    # For first step, block enemy tiles too
                    if depth == 0 and block_enemies and npos in enemy_positions:
                        continue
                else:
                    npos = pos  # WAIT stays in place

                next_depth = depth + 1

                # Check if npos is safe to be at time next_depth
                h = hazard.get(npos, INF_TIMER)
                if h <= next_depth:
                    continue  # Would get hit by explosion

                key = (npos, next_depth)
                if key in seen:
                    continue
                seen.add(key)

                fa = action if first_action == -1 else first_action
                queue.append((npos, next_depth, fa))

        # If blocking enemies failed, try without blocking
        if block_enemies:
            return self._escape_action(grid, my_pos, hazard, bomb_positions,
                                       enemy_positions, H, W, max_depth,
                                       require_no_hazard, block_enemies=False)
        # Last resort: try relaxed safety (safe with margin instead of no hazard)
        if require_no_hazard:
            return self._escape_action(grid, my_pos, hazard, bomb_positions,
                                       enemy_positions, H, W, max_depth,
                                       require_no_hazard=False, block_enemies=False)
        return None

    def _desperation_move(self, grid, my_pos, hazard, bomb_positions, H, W):
        """When no escape found, move to highest-timer neighbor."""
        best_act = 0
        best_timer = hazard.get(my_pos, 0)
        for action in MOVE_ACTIONS:
            dx, dy = MOVES[action]
            nx, ny = my_pos[0] + dx, my_pos[1] + dy
            npos = (nx, ny)
            if not self._passable(grid, nx, ny, H, W):
                continue
            if npos in bomb_positions and npos != my_pos:
                continue
            t = hazard.get(npos, INF_TIMER)
            if t > best_timer:
                best_timer = t
                best_act = action
        return best_act

    # ═══════════════════════════════════════════════════════
    # CAN ESCAPE AFTER PLACING BOMB
    # ═══════════════════════════════════════════════════════

    def _can_escape_after_bomb(self, grid, my_pos, my_radius, bomb_list,
                                bomb_positions, enemy_positions, players, H, W):
        """Check if we can escape after placing a bomb at my_pos.
        Uses timer=6 (safety buffer) and blocks enemy positions."""
        # Create hypothetical bomb (timer=6 for safety margin)
        extra = ((my_pos[0], my_pos[1], 6, self.agent_id, my_radius),)
        new_hazard = self._build_hazard(grid, bomb_list, H, W, extra_bombs=extra)
        new_bomb_pos = bomb_positions | {my_pos}

        # Block enemy positions (they could move to block us)
        blocked_enemies = set()
        for ex, ey, _ in [e for e in [] if False]:  # placeholder
            blocked_enemies.add((ex, ey))

        # Try escape with temporal BFS
        action = self._escape_action(
            grid, my_pos, new_hazard, new_bomb_pos,
            enemy_positions, H, W, max_depth=12,
            require_no_hazard=True, block_enemies=True
        )
        return action is not None

    def _count_safe_reachable(self, grid, my_pos, hazard, bomb_positions, H, W, max_depth=7):
        """Count tiles with no hazard reachable via temporal BFS."""
        queue = deque()
        queue.append((my_pos, 0))
        seen = {(my_pos, 0)}
        safe_tiles = set()

        while queue:
            pos, depth = queue.popleft()
            if pos not in hazard:
                safe_tiles.add(pos)
            if depth >= max_depth:
                continue
            for dx, dy in DIRS:
                npos = (pos[0] + dx, pos[1] + dy)
                if not self._passable(grid, npos[0], npos[1], H, W):
                    continue
                if npos in bomb_positions and npos != pos:
                    continue
                nd = depth + 1
                h = hazard.get(npos, INF_TIMER)
                if h <= nd:
                    continue
                key = (npos, nd)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((npos, nd))

        return len(safe_tiles)

    # ═══════════════════════════════════════════════════════
    # BOMB VALUE EVALUATION
    # ═══════════════════════════════════════════════════════

    def _evaluate_bomb(self, grid, my_pos, my_radius, bomb_list,
                       bomb_positions, enemies, enemy_positions,
                       hazard, players, H, W):
        """Score the value of placing a bomb at my_pos. Returns numerical score."""
        mx, my = my_pos

        # Safety check first (most important)
        extra = ((mx, my, 6, self.agent_id, my_radius),)
        new_hazard = self._build_hazard(grid, bomb_list, H, W, extra_bombs=extra)
        new_bomb_pos = bomb_positions | {my_pos}

        # Temporal escape check with enemy blocking
        escape_act = self._escape_action(
            grid, my_pos, new_hazard, new_bomb_pos,
            enemy_positions, H, W, max_depth=12,
            require_no_hazard=True, block_enemies=True
        )
        if escape_act is None:
            return -INF_TIMER

        # Count safe reachable tiles after bombing
        safe_count = self._count_safe_reachable(
            grid, my_pos, new_hazard, new_bomb_pos, H, W
        )

        # Require minimum safe space (more when enemies close)
        min_enemy_dist = INF_TIMER
        for ex, ey, _ in enemies:
            d = abs(ex - mx) + abs(ey - my)
            if d < min_enemy_dist:
                min_enemy_dist = d

        if min_enemy_dist <= 2:
            required_safe = 5
        elif min_enemy_dist <= 4:
            required_safe = 4
        else:
            required_safe = 3

        if safe_count < required_safe:
            return -INF_TIMER

        # Check escape corridor isn't a dead-end
        esc_dx, esc_dy = MOVES[escape_act]
        esc_tile = (mx + esc_dx, my + esc_dy)
        if escape_act != 0:  # not WAIT
            esc_exits = self._count_open_exits(grid, esc_tile[0], esc_tile[1],
                                                new_bomb_pos, H, W)
            if esc_exits == 0:
                return -INF_TIMER

        # Check open neighbors before bombing (avoid dead-end bombing)
        open_nb = self._count_open_exits(grid, mx, my, bomb_positions, H, W)
        if open_nb <= 1:
            return -INF_TIMER

        # Redundant escape: if enemy near our escape path, need alternative
        if min_enemy_dist <= 3 and escape_act != 0:
            for ex, ey, _ in enemies:
                if abs(ex - esc_tile[0]) + abs(ey - esc_tile[1]) <= 1:
                    # Enemy can block our escape! Need alternative
                    alt_bomb_pos = new_bomb_pos
                    alt_enemy_pos = enemy_positions | {esc_tile}
                    alt_esc = self._escape_action(
                        grid, my_pos, new_hazard, alt_bomb_pos,
                        alt_enemy_pos, H, W, max_depth=10,
                        require_no_hazard=True, block_enemies=False
                    )
                    if alt_esc is None:
                        return -INF_TIMER
                    break

        # ── Now compute value ──────────────────────────────
        blast = set(self._blast_tiles(grid, mx, my, my_radius, H, W))
        value = 0

        # Boxes
        boxes = sum(1 for tile in blast if int(grid[tile[0], tile[1]]) == BOX)
        box_value = 170 if self.step < 400 else 240
        value += boxes * box_value

        # Items destroyed (negative)
        items_destroyed = sum(
            1 for tile in blast
            if int(grid[tile[0], tile[1]]) in (ITEM_R, ITEM_C)
        )
        value -= items_destroyed * 180

        # Enemy hits
        direct_hits = 0
        trapped_hits = 0
        for ex, ey, eid in enemies:
            epos = (ex, ey)
            if epos in blast and self._line_clear(grid, my_pos, epos):
                direct_hits += 1
                # Count enemy escape routes
                enemy_exits = self._enemy_escape_count(
                    grid, epos, blast, bomb_positions, hazard, H, W
                )
                if enemy_exits <= 1:
                    trapped_hits += 1

        value += direct_hits * 600
        value += trapped_hits * 700  # Extra for trapped enemies (total 1300)

        # Check if we trap an enemy completely (guaranteed kill)
        for ex, ey, eid in enemies:
            epos = (ex, ey)
            if self._enemy_trapped(grid, epos, new_hazard, new_bomb_pos, H, W):
                value += 1500  # Guaranteed kill bonus

        # Small bonus for radius upgrade farming
        if my_radius <= 2 and boxes >= 1:
            value += 90

        # Late-game aggression bonus
        if self.step > 430 and boxes == 0 and direct_hits == 0:
            value += 40  # Small bomb for pressure

        # Anti-camping: if late game with 0 kills, be more aggressive
        if self.step > 350 and self._kills == 0:
            value += 80

        # Center preference for bombing (better coverage)
        center_x, center_y = H // 2, W // 2
        dist_to_center = abs(mx - center_x) + abs(my - center_y)
        if dist_to_center <= 3:
            value += 25

        return value

    def _enemy_escape_count(self, grid, enemy_pos, blast, bomb_positions,
                            hazard, H, W):
        """Count how many safe moves an enemy has away from our blast."""
        count = 0
        for dx, dy in DIRS:
            npos = (enemy_pos[0] + dx, enemy_pos[1] + dy)
            if not self._passable(grid, npos[0], npos[1], H, W):
                continue
            if npos in bomb_positions:
                continue
            if npos in blast:
                continue
            if hazard.get(npos, INF_TIMER) <= 1:
                continue
            count += 1
        return count

    def _enemy_trapped(self, grid, epos, hazard, bomb_positions, H, W):
        """BFS check if enemy has 0 safe tiles reachable (guaranteed kill)."""
        queue = deque([epos])
        seen = {epos}
        while queue:
            pos = queue.popleft()
            if pos not in hazard:
                return False  # Enemy can reach a safe tile
            for dx, dy in DIRS:
                npos = (pos[0] + dx, pos[1] + dy)
                if npos in seen:
                    continue
                if not self._passable(grid, npos[0], npos[1], H, W):
                    continue
                if npos in bomb_positions:
                    continue
                seen.add(npos)
                queue.append(npos)
        return True  # No safe tile reachable

    def _bomb_threshold(self):
        """Minimum bomb value to place. Lower = more aggressive."""
        if self.step > 430:
            return 100
        if self.step > 350:
            return 150
        return 200

    # ═══════════════════════════════════════════════════════
    # MOVEMENT TARGET SCORING
    # ═══════════════════════════════════════════════════════

    def _find_move_targets(self, grid, my_pos, my_radius, bomb_positions,
                           enemies, enemy_positions, hazard, bombs_left,
                           bonus, H, W):
        """Score all potential movement targets. Returns dict of {pos: score}."""
        targets = {}
        mx, my = my_pos

        # ── Items ──────────────────────────────────────────
        for x in range(1, H - 1):
            for y in range(1, W - 1):
                cell = int(grid[x, y])
                pos = (x, y)
                if cell == ITEM_C:
                    score = 520 if bombs_left <= 1 else 340
                    targets[pos] = max(targets.get(pos, -INF_TIMER), score)
                elif cell == ITEM_R:
                    score = 500 if bonus < 3 else 300
                    targets[pos] = max(targets.get(pos, -INF_TIMER), score)

        # ── Farm spots (adjacent to boxes) ─────────────────
        for x in range(1, H - 1):
            for y in range(1, W - 1):
                pos = (x, y)
                if not self._passable(grid, x, y, H, W):
                    continue
                if pos in bomb_positions:
                    continue
                box_count = self._count_boxes_in_blast(grid, pos, my_radius, H, W)
                if box_count <= 0:
                    continue
                score = box_count * (180 if self.step > 360 else 120)
                if hazard.get(pos, INF_TIMER) <= 3:
                    score -= 250
                targets[pos] = max(targets.get(pos, -INF_TIMER), score)

        # ── Enemy pressure (mid/late game) ─────────────────
        if enemies and self.step > 60:
            for ex, ey, eid in enemies:
                epos = (ex, ey)
                # Target tiles near enemies
                for action in ALL_ACTIONS:
                    dx, dy = MOVES[action]
                    pos = (ex + dx, ey + dy)
                    if not self._passable(grid, pos[0], pos[1], H, W):
                        continue
                    if pos in bomb_positions:
                        continue
                    dist = abs(pos[0] - mx) + abs(pos[1] - my)
                    score = 180 - dist * 8
                    # Bonus if enemy has few exits (vulnerable)
                    e_exits = self._count_open_exits(grid, ex, ey,
                                                     bomb_positions, H, W)
                    if e_exits <= 2:
                        score += 150  # Enemy in corridor - attack!
                    if self.step > 330:
                        score += 80  # Late game urgency
                    if self._kills == 0 and self.step > 300:
                        score += 100  # Need kills for tiebreak
                    targets[pos] = max(targets.get(pos, -INF_TIMER), score)

        return targets

    def _count_boxes_in_blast(self, grid, pos, radius, H, W):
        """Count boxes that would be hit by a bomb at pos."""
        count = 0
        for tile in self._blast_tiles(grid, pos[0], pos[1], radius, H, W):
            if int(grid[tile[0], tile[1]]) == BOX:
                count += 1
        return count

    # ═══════════════════════════════════════════════════════
    # DANGER-AWARE PATHFINDING
    # ═══════════════════════════════════════════════════════

    def _route_to_scored_targets(self, grid, my_pos, target_scores,
                                  hazard, bomb_positions, H, W,
                                  max_depth=24):
        """BFS to find the best first action toward highest-value reachable target."""
        queue = deque()
        queue.append((my_pos, 0, -1))  # (pos, depth, first_action)
        seen = {my_pos}
        best_action = None
        best_score = -INF_TIMER

        while queue:
            pos, depth, first_action = queue.popleft()

            # Score this position if it's a target
            if pos in target_scores and first_action != -1:
                score = target_scores[pos]
                score -= depth * 18  # Distance penalty
                score += self._local_space_score(grid, pos, bomb_positions,
                                                  hazard, H, W)
                # Penalize if we'd arrive into hazard
                h = hazard.get(pos, INF_TIMER)
                if h <= depth + 1:
                    score -= 1000
                if score > best_score:
                    best_score = score
                    best_action = first_action

            if depth >= max_depth:
                continue

            for action in MOVE_ACTIONS:
                dx, dy = MOVES[action]
                npos = (pos[0] + dx, pos[1] + dy)
                if npos in seen:
                    continue
                if not self._can_enter(grid, npos, bomb_positions, pos, H, W):
                    continue
                nd = depth + 1
                # Safety check: don't walk into danger
                h = hazard.get(npos, INF_TIMER)
                if h <= nd + 1:  # Need buffer of 1
                    continue
                seen.add(npos)
                fa = action if first_action == -1 else first_action
                queue.append((npos, nd, fa))

        return best_action

    # ═══════════════════════════════════════════════════════
    # SAFE IDLE / FALLBACK
    # ═══════════════════════════════════════════════════════

    def _safe_idle_action(self, grid, my_pos, hazard, bomb_positions,
                           enemies, H, W):
        """When no good target, pick the safest/most-open direction."""
        best_action = 0
        best_score = -INF_TIMER

        for action in (0, 1, 2, 3, 4):
            dx, dy = MOVES[action]
            npos = (my_pos[0] + dx, my_pos[1] + dy)

            if action != 0:
                if not self._can_enter(grid, npos, bomb_positions, my_pos, H, W):
                    continue
            else:
                npos = my_pos

            # Must be safe at time step 1
            h = hazard.get(npos, INF_TIMER)
            if h <= 1:
                continue

            score = self._local_space_score(grid, npos, bomb_positions,
                                             hazard, H, W)

            # Distance to enemies
            if enemies:
                nearest = min(abs(npos[0] - ex) + abs(npos[1] - ey)
                              for ex, ey, _ in enemies)
                if nearest <= 2:
                    score -= (3 - nearest) * 35  # Don't get too close
                elif nearest <= 5 and self.step > 330:
                    score += 20  # Approach in late game

            # Prefer movement over standing still
            if action == 0:
                score -= 8

            # Prefer center (more escape routes)
            center_x, center_y = H // 2, W // 2
            dist_c = abs(npos[0] - center_x) + abs(npos[1] - center_y)
            score -= dist_c * 2

            if score > best_score:
                best_score = score
                best_action = action

        return best_action
