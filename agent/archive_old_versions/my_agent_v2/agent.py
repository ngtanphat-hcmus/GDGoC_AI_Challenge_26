"""
AlphaSearch Agent V2 — Hybrid Rule + Tactical Agent
GDGoC-HCMUS AI Challenge 2026 (Bomberland)

V2 improvements over V1:
  1. Timer-aware danger map with chain-reaction propagation
  2. Timed escape BFS (never self-destructs)
  3. Smart bomb placement with trapping detection
  4. Danger-aware pathfinding for ALL movement
  5. Phase-based strategy (early / mid / late game)
  6. Line-of-sight check for enemy bomb hits
  7. Robust fallback (exception → STOP)
"""
import time
import numpy as np
from collections import deque

# ─── Map constants ─────────────────────────────────────────
GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4
PASSABLE = (GRASS, ITEM_R, ITEM_C)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACT_D = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
SAFE = 99            # danger_map value meaning "no bomb here"
MAX_TIMER = 7


# ═══════════════════════════════════════════════════════════
# Module-level helpers (called very frequently; avoid method overhead)
# ═══════════════════════════════════════════════════════════

def _ok(grid, x, y):
    H, W = grid.shape
    return 0 <= x < H and 0 <= y < W and grid[x, y] in PASSABLE


def _blast(grid, bx, by, rad):
    """Return list of (x,y) tiles hit by bomb at (bx,by) with given radius."""
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
    """
    Build danger_map[x,y] = minimum steps until (x,y) is hit by an explosion.
    Chain reactions are propagated: if bomb A's blast reaches bomb B and
    A.timer < B.timer, then B effectively inherits A's timer.
    Value SAFE (99) means no bomb threatens this tile.
    """
    H, W = grid.shape
    dm = np.full((H, W), SAFE, dtype=np.int32)
    if not bombs:
        return dm

    n = len(bombs)
    timers = [b[2] for b in bombs]
    blasts = [set(_blast(grid, b[0], b[1], b[4])) for b in bombs]
    bpos = [(b[0], b[1]) for b in bombs]

    # Fixed-point chain-reaction propagation
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
    """
    Timed-escape BFS.  Find the first action that leads to a tile with
    dm >= SAFE *before* any bomb on the path explodes.

    At BFS depth d we arrive at step d+1.
    We may enter (nx,ny) only if dm[nx,ny] > d+1.
    """
    vis = {start}
    q = deque()
    for a, (dx, dy) in ACT_D.items():
        nx, ny = start[0] + dx, start[1] + dy
        p = (nx, ny)
        if not _ok(grid, nx, ny) or p in blk or p in vis:
            continue
        if dm[nx, ny] <= 1:          # explodes this step or earlier
            continue
        vis.add(p)
        if dm[nx, ny] >= SAFE:       # destination is bomb-free
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


def _navigate(start, targets, grid, dm, blk, max_d=25):
    """Danger-aware BFS to nearest target.  Avoids tiles with dm <= 2."""
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
    """Count how many safe tiles (dm>=SAFE) the enemy can reach.  Early-exit ≥3."""
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
    """True if no wall/box blocks the line from (ax,ay) to (bx,by) exclusive."""
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
    """Number of passable neighbours (0-4)."""
    return sum(1 for dx, dy in DIRS if _ok(grid, x + dx, y + dy))


# ═══════════════════════════════════════════════════════════
# Agent class
# ═══════════════════════════════════════════════════════════

class Agent:
    team_id = "AlphaSearchV2"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0

    # ─── public entry point ────────────────────────────────
    def act(self, obs):
        try:
            return self._decide(obs)
        except Exception:
            return 0

    # ─── main decision loop ────────────────────────────────
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

        # ── parse bombs ────────────────────────────────────
        bombs, bset = [], set()
        for b in raw_bombs:
            bx, by, tm, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            rd = max(1, int(players[oid][4]) + 1) if 0 <= oid < len(players) else 2
            bombs.append((bx, by, tm, oid, rd))
            bset.add((bx, by))

        # ── parse enemies ──────────────────────────────────
        enemies, eset = [], set()
        for i, p in enumerate(players):
            if i != self.agent_id and p[2] == 1:
                enemies.append((int(p[0]), int(p[1])))
                eset.add((int(p[0]), int(p[1])))

        dm = _danger_map(grid, bombs)
        blk = (bset | eset) - {mpos}

        # ═══ 1 — ESCAPE ═══════════════════════════════════
        if dm[mx, my] < SAFE:
            act = _escape(mpos, dm, grid, blk)
            if act is not None:
                return act
            # desperation: least-dangerous neighbour
            ba, bd = 0, dm[mx, my]
            for a, (dx, dy) in ACT_D.items():
                nx, ny = mx + dx, my + dy
                if _ok(grid, nx, ny) and (nx, ny) not in blk and dm[nx, ny] > bd:
                    bd = dm[nx, ny]; ba = a
            return ba

        # ═══ 2 — PLACE BOMB ═══════════════════════════════
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

        # ═══ 4 — MOVE TO BOX-ADJACENT SPOTS ══════════════
        bspots = set()
        for x in range(H):
            for y in range(W):
                if grid[x, y] == BOX:
                    for dx, dy in DIRS:
                        nx, ny = x + dx, y + dy
                        if _ok(grid, nx, ny) and (nx, ny) not in blk:
                            bspots.add((nx, ny))
        if bspots:
            mv = _navigate(mpos, bspots, grid, dm, blk)
            if mv is not None:
                return mv

        # ═══ 5 — PRESSURE NEAREST ENEMY ══════════════════
        if enemies:
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
                    bs = s; ba = a
        return ba

    # ─── bomb evaluation ───────────────────────────────────
    def _should_bomb(self, grid, mpos, bombs, bset, radius,
                     enemies, eset, dm, blk):
        """Return True if placing a bomb here is both valuable AND safe."""
        mx, my = mpos
        my_bl = set(_blast(grid, mx, my, radius))

        # ── value check ────────────────────────────────────
        boxes = sum(1 for x, y in my_bl if grid[x, y] == BOX)

        # enemies in direct line-of-sight within blast radius
        e_hit = []
        for ex, ey in enemies:
            if (ex, ey) in my_bl:
                if mx == ex or my == ey:     # same row or col
                    if _line_clear(grid, mx, my, ex, ey):
                        e_hit.append((ex, ey))

        # Phase-dependent threshold
        if self.step < 60:
            # Early game: conservative — need good value
            if not e_hit and boxes < 2:
                return False
        elif self.step < 400:
            # Mid game: bomb if decent value
            if not e_hit and boxes < 1:
                return False
        else:
            # Late game: aggressive for tie-break stats
            if not e_hit and boxes < 1:
                return False

        # ── safety check (can we escape after placing?) ────
        new_bombs = list(bombs) + [(mx, my, MAX_TIMER, self.agent_id, radius)]
        new_bset = bset | {mpos}
        new_dm = _danger_map(grid, new_bombs)
        esc_blk = (new_bset | eset) - {mpos}

        if _escape(mpos, new_dm, grid, esc_blk) is None:
            return False

        # ── trapping bonus (not required, but prefer kills) ─
        # If any enemy would be trapped (0 safe tiles), definitely bomb
        for ex, ey in enemies:
            if _enemy_safe(grid, (ex, ey), new_dm, new_bset) == 0:
                return True   # guaranteed kill — always worth it

        # ── accept if we have any value ────────────────────
        return True
