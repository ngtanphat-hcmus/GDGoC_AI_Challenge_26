"""
AlphaSearch Agent V3 — Anti-Self-Destruct + Tactical Aggression
GDGoC-HCMUS AI Challenge 2026 (Bomberland)

V3 fixes from V2 death analysis (8/11 losses = SELF_BOMB):
  1. Multi-layer bomb safety: basic escape + safe reachable count + enemy block check
  2. Corridor detection: never bomb with <= 1 open neighbor
  3. Enemy proximity penalty: stricter when enemies within 3 tiles
  4. Escape fallback: try walking through enemies if standard escape fails
  5. Better farming: prefer box spots with more boxes in blast
  6. Adaptive aggression: hunt harder when powered up
"""
import time
import numpy as np
from collections import deque

# ─── Constants ─────────────────────────────────────────────
GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4
PASSABLE = (GRASS, ITEM_R, ITEM_C)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACT_D = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
SAFE = 99
MAX_TIMER = 7


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
    H, W = grid.shape
    dm = np.full((H, W), SAFE, dtype=np.int32)
    if not bombs:
        return dm
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
            if t < dm[x, y]:
                dm[x, y] = t
    return dm


def _escape(start, dm, grid, blk, max_d=10):
    """Timed-escape BFS to a safe tile (dm >= SAFE)."""
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


def _count_safe_reachable(start, dm, grid, blk, max_d=7):
    """Count safe tiles (dm >= SAFE) reachable within max_d steps."""
    vis = {start}
    q = deque([(start[0], start[1], 0)])
    cnt = 0
    while q:
        x, y, d = q.popleft()
        if dm[x, y] >= SAFE:
            cnt += 1
        if d >= max_d:
            continue
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in vis or not _ok(grid, nx, ny) or p in blk:
                continue
            if dm[nx, ny] <= d + 1:
                continue
            vis.add(p)
            q.append((nx, ny, d + 1))
    return cnt


def _navigate(start, targets, grid, dm, blk, max_d=25):
    """Danger-aware BFS to nearest target."""
    if not targets:
        return None
    ts = targets if isinstance(targets, set) else set(targets)
    vis = {start}
    q = deque()
    for a, (dx, dy) in ACT_D.items():
        nx, ny = start[0] + dx, start[1] + dy
        p = (nx, ny)
        if p in vis or not _ok(grid, nx, ny):
            continue
        if p in blk and p not in ts:
            continue
        if dm[nx, ny] <= 2:
            continue
        vis.add(p)
        if p in ts:
            return a
        q.append((nx, ny, 1, a))
    while q:
        x, y, d, fa = q.popleft()
        if d >= max_d:
            continue
        for _, (dx, dy) in ACT_D.items():
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in vis or not _ok(grid, nx, ny):
                continue
            if p in blk and p not in ts:
                continue
            if dm[nx, ny] <= 2:
                continue
            vis.add(p)
            if p in ts:
                return fa
            q.append((nx, ny, d + 1, fa))
    return None


def _enemy_safe(grid, epos, dm, bset):
    q = deque([epos])
    vis = {epos}
    cnt = 0
    while q:
        x, y = q.popleft()
        if dm[x, y] >= SAFE:
            cnt += 1
            if cnt >= 3:
                return cnt
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            p = (nx, ny)
            if p in vis or not _ok(grid, nx, ny) or p in bset:
                continue
            vis.add(p)
            q.append(p)
    return cnt


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


def _count_open(grid, x, y):
    return sum(1 for dx, dy in DIRS if _ok(grid, x + dx, y + dy))


def _boxes_from(grid, x, y, rad):
    """Count boxes hit by a bomb placed at (x,y)."""
    return sum(1 for bx, by in _blast(grid, x, y, rad) if grid[bx, by] == BOX)


class Agent:
    team_id = "AlphaSearchV3"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0

    def act(self, obs):
        try:
            return self._decide(obs)
        except Exception:
            return 0

    def _decide(self, obs):
        self.step += 1
        grid = obs["map"]
        players = obs["players"]
        raw_bombs = obs["bombs"]
        H, W = grid.shape

        me = players[self.agent_id]
        if me[2] != 1:
            return 0

        mx, my = int(me[0]), int(me[1])
        mpos = (mx, my)
        bombs_left = int(me[3])
        bonus = int(me[4])
        radius = max(1, bonus + 1)

        bombs, bset = [], set()
        for b in raw_bombs:
            bx, by, tm, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rd = max(1, int(players[oid][4]) + 1) if 0 <= oid < len(players) else 2
            bombs.append((bx, by, tm, oid, rd))
            bset.add((bx, by))

        enemies, eset = [], set()
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                enemies.append((int(p[0]), int(p[1])))
                eset.add((int(p[0]), int(p[1])))

        dm = _danger_map(grid, bombs)
        blk = (bset | eset) - {mpos}

        # ═══ 1 — ESCAPE (multi-fallback) ══════════════════
        if dm[mx, my] < SAFE:
            # Layer 1: standard timed escape
            esc = _escape(mpos, dm, grid, blk)
            if esc is not None:
                return esc

            # Layer 2: escape ignoring enemies (walk through them if needed)
            esc = _escape(mpos, dm, grid, bset - {mpos})
            if esc is not None:
                return esc

            # Layer 3: move to least-dangerous neighbor (ignore enemies)
            ba, bd = 0, dm[mx, my]
            for a, (dx, dy) in ACT_D.items():
                nx, ny = mx + dx, my + dy
                if _ok(grid, nx, ny) and (nx, ny) not in bset:
                    if dm[nx, ny] > bd:
                        bd = dm[nx, ny]
                        ba = a
            return ba

        # ═══ 2 — PLACE BOMB (multi-layer safety) ══════════
        if bombs_left > 0 and mpos not in bset:
            if self._should_bomb(grid, mpos, bombs, bset, radius,
                                 enemies, eset, dm, blk):
                return 5

        # ═══ 3 — COLLECT ITEMS ════════════════════════════
        items, pref = set(), set()
        for x in range(H):
            for y in range(W):
                c = grid[x, y]
                if c == ITEM_R:
                    items.add((x, y))
                    if bonus < 3:
                        pref.add((x, y))
                elif c == ITEM_C:
                    items.add((x, y))
                    if bombs_left <= 1:
                        pref.add((x, y))
        tgt = pref or items
        if tgt:
            mv = _navigate(mpos, tgt, grid, dm, blk)
            if mv is not None:
                return mv

        # ═══ 4 — MOVE TO BEST BOX SPOTS ══════════════════
        bspots = set()
        for x in range(H):
            for y in range(W):
                if grid[x, y] == BOX:
                    for dx, dy in DIRS:
                        nx, ny = x + dx, y + dy
                        if _ok(grid, nx, ny) and (nx, ny) not in blk:
                            bspots.add((nx, ny))

        # Prefer spots that hit more boxes
        if bspots:
            best_spots = set()
            best_count = 0
            for sp in bspots:
                bc = _boxes_from(grid, sp[0], sp[1], radius)
                if bc > best_count:
                    best_count = bc
                    best_spots = {sp}
                elif bc == best_count:
                    best_spots.add(sp)

            mv = _navigate(mpos, best_spots, grid, dm, blk)
            if mv is not None:
                return mv
            # Fallback: any box spot
            mv = _navigate(mpos, bspots, grid, dm, blk)
            if mv is not None:
                return mv

        # ═══ 5 — PRESSURE ENEMIES ════════════════════════
        if enemies:
            # When powered up, be more aggressive — move closer
            mv = _navigate(mpos, eset, grid, dm, blk)
            if mv is not None:
                return mv

        # ═══ 6 — SAFE FALLBACK ═══════════════════════════
        ba, bs = 0, -1
        for a, (dx, dy) in ACT_D.items():
            nx, ny = mx + dx, my + dy
            if _ok(grid, nx, ny) and (nx, ny) not in blk and dm[nx, ny] >= SAFE:
                s = _count_open(grid, nx, ny)
                if s > bs:
                    bs = s
                    ba = a
        return ba

    def _should_bomb(self, grid, mpos, bombs, bset, radius,
                     enemies, eset, dm, blk):
        """Multi-layer bomb safety + value check."""
        mx, my = mpos
        my_blast = set(_blast(grid, mx, my, radius))

        # ─── VALUE CHECK ──────────────────────────────────
        boxes = sum(1 for x, y in my_blast if grid[x, y] == BOX)
        e_hit = []
        for ex, ey in enemies:
            if (ex, ey) in my_blast:
                if mx == ex or my == ey:
                    if _line_clear(grid, mx, my, ex, ey):
                        e_hit.append((ex, ey))

        # Phase-dependent thresholds
        if self.step < 80:
            if not e_hit and boxes < 2:
                return False
        elif self.step < 400:
            if not e_hit and boxes < 1:
                return False
        else:
            if not e_hit and boxes < 1:
                return False

        # ─── LAYER 0: Open neighbor check ─────────────────
        # Don't bomb if we have <= 1 open neighbor (dead-end / tight corridor)
        open_nb = 0
        for dx, dy in DIRS:
            nx, ny = mx + dx, my + dy
            if _ok(grid, nx, ny) and (nx, ny) not in bset:
                open_nb += 1
        if open_nb <= 1:
            return False

        # ─── LAYER 1: Build post-bomb danger map ──────────
        new_bombs = list(bombs) + [(mx, my, MAX_TIMER, self.agent_id, radius)]
        new_bset = bset | {mpos}
        new_dm = _danger_map(grid, new_bombs)
        esc_blk = (new_bset | eset) - {mpos}

        # ─── LAYER 2: Basic escape check ──────────────────
        esc_action = _escape(mpos, new_dm, grid, esc_blk)
        if esc_action is None:
            return False

        # ─── LAYER 3: Count safe reachable tiles ──────────
        # Must have enough room to maneuver even if situation changes
        min_enemy_dist = 99
        for ex, ey in enemies:
            d = abs(ex - mx) + abs(ey - my)
            if d < min_enemy_dist:
                min_enemy_dist = d

        # Require more safe space when enemies are close
        if min_enemy_dist <= 2:
            required_safe = 5
        elif min_enemy_dist <= 4:
            required_safe = 4
        else:
            required_safe = 3

        safe_count = _count_safe_reachable(mpos, new_dm, grid, esc_blk)
        if safe_count < required_safe:
            return False

        # ─── LAYER 4: Redundant escape paths ──────────────
        # If enemy is close, verify escape works even if enemy blocks first route
        if min_enemy_dist <= 3 and esc_action is not None:
            esc_dx, esc_dy = ACT_D[esc_action]
            first_tile = (mx + esc_dx, my + esc_dy)
            # Check if any enemy could reach our first escape tile
            for ex, ey in enemies:
                if abs(ex - first_tile[0]) + abs(ey - first_tile[1]) <= 1:
                    # Enemy adjacent to our escape! Need alternative path
                    alt_blk = esc_blk | {first_tile}
                    alt_esc = _escape(mpos, new_dm, grid, alt_blk)
                    if alt_esc is None:
                        return False  # Only 1 escape route, enemy can block it
                    break

        # ─── LAYER 5: Don't bomb if we'd block ourselves in a corridor ───
        # After bombing, check if our escape path goes through a narrow section
        # (the escape tile itself has <= 1 open neighbor besides us)
        esc_dx, esc_dy = ACT_D[esc_action]
        esc_tile = (mx + esc_dx, my + esc_dy)
        esc_exits = 0
        for dx, dy in DIRS:
            nx, ny = esc_tile[0] + dx, esc_tile[1] + dy
            if (nx, ny) != mpos and _ok(grid, nx, ny) and (nx, ny) not in new_bset:
                esc_exits += 1
        if esc_exits == 0:
            return False  # Escape tile is a dead-end!

        # ─── BONUS: Trapping detection ────────────────────
        for ex, ey in enemies:
            if _enemy_safe(grid, (ex, ey), new_dm, new_bset) == 0:
                return True  # Guaranteed kill

        return True
