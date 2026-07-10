"""
AlphaSearch Agent V7 — Competition-Ready Bomberland AI
GDGoC-HCMUS AI Challenge 2026

Built on V6 + advanced competitive features:
  1. Enemy-Aware Target Selection: ignores items/boxes that enemies can reach faster.
  2. 1-Step Minimax Threat Modeling: simulates potential enemy bomb placements in close combat.
  3. Optimized Escape Logic: accounts for immediate danger and potential enemy threat zones.
  4. Chase-Defense & Choke-Point Bombing.
  5. Swapped bombing & escape phases, with safe temporal depth=1 validation.
"""
import numpy as np
from collections import deque

# ─── Constants ─────────────────────────────────────────────
GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4
PASSABLE = (GRASS, ITEM_R, ITEM_C)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACT_D = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
ALL_ACT = (1, 2, 3, 4, 0)   # includes WAIT for escape
SAFE = 99
BOMB_TIMER = 7
INF = 10**6


def _ok(grid, x, y):
    H, W = grid.shape
    return 0 <= x < H and 0 <= y < W and grid[x, y] in PASSABLE


def _blast(grid, bx, by, rad):
    H, W = grid.shape
    out = [(bx, by)]
    for dx, dy in DIRS:
        for r in range(1, rad + 1):
            x, y = bx + dx * r, by + dy * r
            if x < 0 or x >= H or y < 0 or y >= W:
                break
            c = grid[x, y]
            if c == WALL:
                break
            out.append((x, y))
            if c == BOX:
                break
    return out


def _danger_map(grid, bombs):
    """Build danger map: dm[x,y] = min timer of any bomb blast reaching (x,y). SAFE=99 means no danger."""
    H, W = grid.shape
    dm = np.full((H, W), SAFE, dtype=np.int32)
    if not bombs:
        return dm
    n = len(bombs)
    timers = [b[2] for b in bombs]
    blasts = [set(_blast(grid, b[0], b[1], b[4])) for b in bombs]
    bpos = [(b[0], b[1]) for b in bombs]
    # Chain reactions: propagate minimum timer
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if i != j and bpos[j] in blasts[i] and timers[j] > timers[i]:
                    timers[j] = timers[i]
                    changed = True
    for i in range(n):
        t = max(1, timers[i])
        for x, y in blasts[i]:
            if t < dm[x, y]:
                dm[x, y] = t
    return dm


def _hazard_dict(grid, bombs):
    """Build hazard dict: pos -> min timer. Only includes tiles with active danger."""
    haz = {}
    if not bombs:
        return haz
    n = len(bombs)
    timers = [b[2] for b in bombs]
    blasts = [set(_blast(grid, b[0], b[1], b[4])) for b in bombs]
    bpos = [(b[0], b[1]) for b in bombs]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if i != j and bpos[j] in blasts[i] and timers[j] > timers[i]:
                    timers[j] = timers[i]
                    changed = True
    for i in range(n):
        t = max(1, timers[i])
        for x, y in blasts[i]:
            if t < haz.get((x, y), INF):
                haz[(x, y)] = t
    return haz


def _temporal_escape(start, hazard, grid, bomb_positions, enemy_positions, max_d=14, start_depth=0):
    """
    Temporal BFS: state = (position, time_step). Includes WAIT action.
    Returns first action to reach a tile with NO future hazard.
    """
    H, W = grid.shape
    seen = {(start, start_depth)}
    q = deque([(start, start_depth, None)])

    while q:
        pos, depth, first_action = q.popleft()
        if depth > start_depth and pos not in hazard:
            return first_action
        if depth >= max_d:
            continue

        for action in ALL_ACT:
            if action == 0:
                npos = pos
            else:
                dx, dy = ACT_D[action]
                npos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= npos[0] < H and 0 <= npos[1] < W):
                    continue
                if grid[npos[0], npos[1]] not in PASSABLE:
                    continue
                if npos in bomb_positions and npos != pos:
                    continue

            next_t = depth + 1
            haz_timer = hazard.get(npos, INF)
            if haz_timer <= next_t:
                continue

            key = (npos, next_t)
            if key in seen:
                continue
            seen.add(key)
            fa = action if first_action is None else first_action
            q.append((npos, next_t, fa))

    return None


def _simple_escape(start, dm, grid, blk, max_d=10):
    vis = {start}
    q = deque()
    for a, (dx, dy) in ACT_D.items():
        nx, ny = start[0] + dx, start[1] + dy
        p = (nx, ny)
        if not _ok(grid, nx, ny) or p in blk or p in vis:
            continue
        if dm[nx, ny] <= 1:
            continue
        vis.add(p)
        if dm[nx, ny] >= SAFE:
            return a
        q.append((nx, ny, 1, a))
    while q:
        x, y, d, fa = q.popleft()
        if d >= max_d:
            continue
        for _, (dx, dy) in ACT_D.items():
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in vis or not _ok(grid, nx, ny) or p in blk:
                continue
            if dm[nx, ny] <= d + 1:
                continue
            vis.add(p)
            if dm[nx, ny] >= SAFE:
                return fa
            q.append((nx, ny, d + 1, fa))
    return None


def _route_scored(start, target_scores, grid, hazard, bomb_positions, max_d=24):
    H, W = grid.shape
    q = deque([(start, 0, None)])
    seen = {start}
    best_action = None
    best_score = -INF

    while q:
        pos, depth, first_action = q.popleft()
        if depth > max_d:
            continue
        if pos in target_scores and first_action is not None:
            score = target_scores[pos] - depth * 18 + _local_space(grid, pos, bomb_positions, hazard)
            ht = hazard.get(pos, INF)
            if ht <= depth + 1:
                score -= 1000
            if score > best_score:
                best_score = score
                best_action = first_action

        if depth >= max_d:
            continue
        for action in (1, 2, 3, 4):
            dx, dy = ACT_D[action]
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in seen:
                continue
            if not (0 <= nx < H and 0 <= ny < W) or grid[nx, ny] not in PASSABLE:
                continue
            if npos in bomb_positions and npos != pos:
                continue
            ht = hazard.get(npos, INF)
            if ht <= depth + 2:
                continue
            seen.add(npos)
            q.append((npos, depth + 1, action if first_action is None else first_action))
    return best_action


def _local_space(grid, pos, bomb_positions, hazard):
    score = 0
    for dx, dy in DIRS:
        nx, ny = pos[0] + dx, pos[1] + dy
        if _ok(grid, nx, ny) and (nx, ny) not in bomb_positions:
            score += 12
    c = grid[pos[0], pos[1]]
    if c == ITEM_C:
        score += 70
    elif c == ITEM_R:
        score += 60
    if hazard.get(pos, INF) <= 3:
        score -= 80
    return score


def _line_clear(grid, ax, ay, bx, by):
    if ax == bx:
        step = 1 if by > ay else -1
        for y in range(ay + step, by, step):
            if grid[ax, y] in (WALL, BOX):
                return False
        return True
    if ay == by:
        step = 1 if bx > ax else -1
        for x in range(ax + step, bx, step):
            if grid[x, ay] in (WALL, BOX):
                return False
        return True
    return False


def _boxes_from(grid, x, y, rad):
    return sum(1 for bx, by in _blast(grid, x, y, rad) if grid[bx, by] == BOX)


def _items_in_blast(grid, x, y, rad):
    return sum(1 for bx, by in _blast(grid, x, y, rad) if grid[bx, by] in (ITEM_R, ITEM_C))


def _simulate_enemy_trapped(grid, mpos, radius, enemy_pos, bombs, players, agent_id):
    """Simulate placing a bomb at mpos and count the enemy's safe escape paths."""
    temp_bombs = list(bombs) + [(mpos[0], mpos[1], 6, agent_id, radius)]
    temp_dm = _danger_map(grid, temp_bombs)
    bset = {(b[0], b[1]) for b in temp_bombs}
    
    q = deque([enemy_pos])
    vis = {enemy_pos}
    cnt = 0
    while q:
        if cnt >= 2:
            return False
        x, y = q.popleft()
        if temp_dm[x, y] >= SAFE:
            cnt += 1
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in vis or not _ok(grid, nx, ny) or p in bset:
                continue
            vis.add(p)
            q.append(p)
            
    return cnt <= 1


class Agent:
    team_id = "AlphaSearchV7"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0
        self._bomb_radii = {}
        self._last_sig = None

    def act(self, obs):
        try:
            return self._decide(obs)
        except Exception:
            return 0

    def _decide(self, obs):
        grid = obs["map"]
        players = obs["players"]
        raw_bombs = obs["bombs"]
        H, W = grid.shape

        # New game detection
        if self._is_new_game(obs, H, W):
            self.step = 0
            self._bomb_radii = {}
        self.step += 1

        me = players[self.agent_id]
        if me[2] != 1:
            return 0

        mx, my = int(me[0]), int(me[1])
        mpos = (mx, my)
        bombs_left = int(me[3])
        bonus = int(me[4])
        radius = max(1, bonus + 1)

        # Sync bomb memory
        self._sync_bomb_memory(raw_bombs, players)

        # Parse bombs with remembered radii
        bombs, bset = [], set()
        for b in raw_bombs:
            bx, by, tm, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rd = self._get_bomb_radius(bx, by, oid, players)
            bombs.append((bx, by, tm, oid, rd))
            bset.add((bx, by))

        enemies, eset = [], set()
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                enemies.append((int(p[0]), int(p[1])))
                eset.add((int(p[0]), int(p[1])))

        # ─── MINIMAX THREAT MODELING (Enemy-potential bomb) ────────
        # If an enemy is close, assume they place a bomb and add it to potential hazards
        pot_bombs = list(bombs)
        closest_enemy_dist = 99
        closest_enemy_idx = -1
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                d = abs(int(p[0]) - mx) + abs(int(p[1]) - my)
                if d < closest_enemy_dist:
                    closest_enemy_dist = d
                    closest_enemy_idx = i

        if closest_enemy_idx != -1 and closest_enemy_dist <= 3:
            ep = players[closest_enemy_idx]
            ex, ey = int(ep[0]), int(ep[1])
            erand = max(1, int(ep[4]) + 1)
            # Add potential enemy bomb with timer=7
            pot_bombs.append((ex, ey, 7, closest_enemy_idx, erand))

        dm = _danger_map(grid, bombs)
        hazard = _hazard_dict(grid, bombs)

        # Potential hazards (including enemy threats)
        pot_dm = _danger_map(grid, pot_bombs)
        pot_hazard = _hazard_dict(grid, pot_bombs)

        blk = (bset | eset) - {mpos}

        # ═══ 1 — IMMEDIATE ESCAPE (Critical threat) ═════════
        # If a bomb is about to explode next step (or we are in critical danger), run immediately!
        if dm[mx, my] <= 2:
            esc = _temporal_escape(mpos, hazard, grid, bset, eset)
            if esc is not None:
                return esc
            esc = _simple_escape(mpos, dm, grid, bset - {mpos})
            if esc is not None:
                return esc

        # ═══ 2 — PLACE BOMB (Temporal escape depth=1 checks pot_hazard) ═
        if bombs_left > 0 and mpos not in bset:
            if self._should_bomb(grid, mpos, radius, bombs, pot_bombs, bset,
                                 enemies, eset, dm, hazard, pot_hazard, blk, players):
                return 5

        # ═══ 3 — TEMPORAL ESCAPE (Using pot_hazard if safe) ═════
        if dm[mx, my] < SAFE:
            # Try escaping using potential threats first
            esc = _temporal_escape(mpos, pot_hazard, grid, bset, eset)
            if esc is not None:
                return esc
            # Fallback to existing danger only
            esc = _temporal_escape(mpos, hazard, grid, bset, eset)
            if esc is not None:
                return esc
            esc = _simple_escape(mpos, dm, grid, bset - {mpos})
            if esc is not None:
                return esc
            # Desperation fallback
            ba, bd = 0, dm[mx, my]
            for a, (dx, dy) in ACT_D.items():
                nx, ny = mx + dx, my + dy
                if _ok(grid, nx, ny) and (nx, ny) not in bset:
                    if dm[nx, ny] > bd:
                        bd = dm[nx, ny]
                        ba = a
            return ba

        # ═══ 4 — COLLECT ITEMS (Enemy-aware target scoring) ════
        item_targets = {}
        for x in range(H):
            for y in range(W):
                c = grid[x, y]
                if c == ITEM_C or c == ITEM_R:
                    d_me = abs(x - mx) + abs(y - my)
                    d_enemy = min(abs(x - ex) + abs(y - ey) for ex, ey in enemies) if enemies else INF
                    if d_enemy < d_me:
                        continue  # Enemy is closer, ignore item
                    
                    if c == ITEM_C:
                        item_targets[(x, y)] = 520 if bombs_left <= 1 else 340
                    else:
                        item_targets[(x, y)] = 500 if bonus < 2 else 300

        if item_targets:
            mv = _route_scored(mpos, item_targets, grid, pot_hazard, bset)
            if mv is not None:
                return mv

        # ═══ 5 — MOVE TO BEST BOX SPOTS (Enemy-aware) ═══════
        farm_targets = {}
        for x in range(1, H - 1):
            for y in range(1, W - 1):
                if not _ok(grid, x, y) or (x, y) in bset:
                    continue
                bc = _boxes_from(grid, x, y, radius)
                if bc <= 0:
                    continue
                
                d_me = abs(x - mx) + abs(y - my)
                d_enemy = min(abs(x - ex) + abs(y - ey) for ex, ey in enemies) if enemies else INF
                if d_enemy < d_me:
                    continue  # Enemy is closer, ignore this spot
                
                score = bc * (180 if self.step > 360 else 120)
                if pot_hazard.get((x, y), INF) <= 3:
                    score -= 250
                farm_targets[(x, y)] = score

        if farm_targets:
            if mpos in farm_targets and bombs_left > 0 and mpos not in bset:
                bv = self._bomb_value(grid, mpos, radius, bombs, bset, enemies, eset, dm, hazard, blk, players)
                if bv >= self._bomb_threshold():
                    return 5

            mv = _route_scored(mpos, farm_targets, grid, pot_hazard, bset)
            if mv is not None:
                return mv

        # ═══ 6 — PRESSURE ENEMIES (scored routing) ══════════
        if enemies:
            pressure_targets = {}
            for ex, ey in enemies:
                for action in (1, 2, 3, 4, 0):
                    if action == 0:
                        pos = (ex, ey)
                    else:
                        dx, dy = ACT_D[action]
                        pos = (ex + dx, ey + dy)
                    if not _ok(grid, pos[0], pos[1]) or pos in bset:
                        continue
                    dist = abs(pos[0] - mx) + abs(pos[1] - my)
                    score = 180 - dist * 8
                    exits = sum(1 for dx, dy in DIRS
                                if _ok(grid, ex + dx, ey + dy) and (ex + dx, ey + dy) not in bset)
                    if exits <= 2:
                        score += 120
                    if self.step > 330:
                        score += 80
                    pressure_targets[pos] = max(pressure_targets.get(pos, -INF), score)

            mv = _route_scored(mpos, pressure_targets, grid, pot_hazard, bset)
            if mv is not None:
                return mv

        # ═══ 7 — SAFE IDLE (prefer open areas) ══════════════
        ba, bs = 0, -INF
        for a in (1, 2, 3, 4, 0):
            if a == 0:
                npos = mpos
            else:
                dx, dy = ACT_D[a]
                npos = (mx + dx, my + dy)
                if not _ok(grid, npos[0], npos[1]) or npos in bset:
                    continue

            ht = pot_hazard.get(npos, INF)
            if ht <= 2:
                continue

            s = _local_space(grid, npos, bset, pot_hazard)
            if enemies:
                nearest = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey in enemies)
                if nearest <= 2:
                    s -= (3 - nearest) * 30
                elif nearest <= 5 and self.step > 330:
                    s += 20
            if a == 0:
                s -= 8
            if s > bs:
                bs = s
                ba = a
        return ba

    # ─── Bomb Value Scoring ───────────────────────────────
    def _bomb_value(self, grid, mpos, radius, bombs, bset, enemies, eset, dm, hazard, blk, players):
        """Score the value of placing a bomb. Returns negative if unsafe."""
        mx, my = mpos
        my_blast = set(_blast(grid, mx, my, radius))

        boxes = sum(1 for x, y in my_blast if grid[x, y] == BOX)
        items_destroyed = _items_in_blast(grid, mx, my, radius)
        e_hit = []
        e_trapped = 0
        for ex, ey in enemies:
            if (ex, ey) in my_blast and (mx == ex or my == ey):
                if _line_clear(grid, mx, my, ex, ey):
                    e_hit.append((ex, ey))
                    exits = sum(1 for dx, dy in DIRS
                                if _ok(grid, ex + dx, ey + dy) and
                                (ex + dx, ey + dy) not in my_blast and
                                (ex + dx, ey + dy) not in bset and
                                hazard.get((ex + dx, ey + dy), INF) > 1)
                    if exits <= 1:
                        e_trapped += 1

        # ─── ADDITION 1: Active Trap Simulation ────────────
        e_trapped_sim = 0
        for ex, ey in enemies:
            dist = abs(ex - mx) + abs(ey - my)
            if dist <= 4:
                if _simulate_enemy_trapped(grid, mpos, radius, (ex, ey), bombs, players, self.agent_id):
                    e_trapped_sim += 1

        # ─── ADDITION 2: Chase-Defense ─────────────────────
        chase_defense = 0
        for ex, ey in enemies:
            dist = abs(ex - mx) + abs(ey - my)
            if dist <= 2:
                chase_defense += 1

        value = 0
        value += len(e_hit) * 600
        value += e_trapped * 700
        value += e_trapped_sim * 800
        value += chase_defense * 300
        value += boxes * (170 if self.step < 400 else 240)
        value -= items_destroyed * 180

        if self.step > 430 and boxes == 0 and not e_hit:
            value += 40
        if radius <= 2 and boxes >= 1:
            value += 90

        return value

    def _bomb_threshold(self):
        if self.step > 430:
            return 130
        if self.step > 350:
            return 150
        return 150  # Lowered from 200 in V5 to enable aggressive early clearing

    def _should_bomb(self, grid, mpos, radius, bombs, pot_bombs, bset,
                     enemies, eset, dm, hazard, pot_hazard, blk, players):
        mx, my = mpos

        # ─── VALUE CHECK ──────────────────────────────────
        bv = self._bomb_value(grid, mpos, radius, bombs, bset, enemies, eset, dm, hazard, blk, players)
        if bv < self._bomb_threshold():
            return False

        # ─── LAYER 0: Open neighbor check ─────────────────
        open_nb = 0
        for dx, dy in DIRS:
            nx, ny = mx + dx, my + dy
            if _ok(grid, nx, ny) and (nx, ny) not in bset:
                open_nb += 1
        if open_nb <= 1:
            return False

        # ─── LAYER 1: Build post-bomb danger maps ──────────
        new_bombs = list(bombs) + [(mx, my, 6, self.agent_id, radius)]
        new_pot_bombs = list(pot_bombs) + [(mx, my, 6, self.agent_id, radius)]
        new_bset = bset | {mpos}
        
        # Build regular and potential danger maps
        new_dm = _danger_map(grid, new_bombs)
        new_hazard = _hazard_dict(grid, new_bombs)
        
        new_pot_dm = _danger_map(grid, new_pot_bombs)
        new_pot_hazard = _hazard_dict(grid, new_pot_bombs)
        
        esc_blk = (new_bset | eset) - {mpos}

        # ─── LAYER 2: Temporal escape check (Using pot_hazard for double-safety) ───
        # Try escaping with potential enemy threats modeled
        esc_action = _temporal_escape(mpos, new_pot_hazard, grid, new_bset, eset, max_d=12, start_depth=1)
        if esc_action is None:
            # Fallback to escaping regular bombs only
            esc_action = _temporal_escape(mpos, new_hazard, grid, new_bset, eset, max_d=12, start_depth=1)
            if esc_action is None:
                return False

        # ─── LAYER 3: safe_count check REMOVED for V6/V7 ──────

        # ─── LAYER 4: Redundant escape paths ──────────────
        min_enemy_dist = 99
        for ex, ey in enemies:
            d = abs(ex - mx) + abs(ey - my)
            if d < min_enemy_dist:
                min_enemy_dist = d

        if min_enemy_dist <= 3 and esc_action in ACT_D:
            esc_dx, esc_dy = ACT_D[esc_action]
            first_tile = (mx + esc_dx, my + esc_dy)
            for ex, ey in enemies:
                if abs(ex - first_tile[0]) + abs(ey - first_tile[1]) <= 1:
                    alt_blk = esc_blk | {first_tile}
                    alt_esc = _simple_escape(mpos, new_dm, grid, alt_blk)
                    if alt_esc is None:
                        return False
                    break

        # ─── LAYER 5: Escape tile dead-end check ──────────
        if esc_action in ACT_D:
            esc_dx, esc_dy = ACT_D[esc_action]
            esc_tile = (mx + esc_dx, my + esc_dy)
            esc_exits = 0
            for dx, dy in DIRS:
                nx, ny = esc_tile[0] + dx, esc_tile[1] + dy
                if (nx, ny) != mpos and _ok(grid, nx, ny) and (nx, ny) not in new_bset:
                    esc_exits += 1
            if esc_exits == 0:
                return False

        return True

    # ─── Bomb Memory ──────────────────────────────────────
    def _sync_bomb_memory(self, raw_bombs, players):
        active = set()
        for b in raw_bombs:
            bx, by, _, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            key = (bx, by, oid)
            active.add(key)
            if key not in self._bomb_radii:
                if 0 <= oid < len(players):
                    self._bomb_radii[key] = max(1, int(players[oid][4]) + 1)
                else:
                    self._bomb_radii[key] = 1
        for key in list(self._bomb_radii):
            if key not in active:
                del self._bomb_radii[key]

    def _get_bomb_radius(self, bx, by, oid, players):
        r = self._bomb_radii.get((bx, by, oid))
        if r is not None:
            return r
        if 0 <= oid < len(players):
            return max(1, int(players[oid][4]) + 1)
        return 1

    # ─── New Game Detection ───────────────────────────────
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
            result = sig != self._last_sig
            self._last_sig = sig
            return result
        except Exception:
            return False

    def _state_sig(self, obs):
        players = obs.get("players", [])
        return tuple((int(p[0]), int(p[1]), int(p[2])) for p in players)
