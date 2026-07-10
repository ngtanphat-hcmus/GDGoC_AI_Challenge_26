from collections import deque


class ParsedState:
    __slots__ = (
        "grid",
        "players",
        "bombs",
        "my_pos",
        "my_bombs_left",
        "my_radius",
        "enemies",
        "bomb_positions",
        "hazard",
        "hazard_now",
        "hazard_soon",
        "step",
    )

    def __init__(
        self,
        grid,
        players,
        bombs,
        my_pos,
        my_bombs_left,
        my_radius,
        enemies,
        bomb_positions,
        hazard,
        hazard_now,
        hazard_soon,
        step,
    ):
        self.grid = grid
        self.players = players
        self.bombs = bombs
        self.my_pos = my_pos
        self.my_bombs_left = my_bombs_left
        self.my_radius = my_radius
        self.enemies = enemies
        self.bomb_positions = bomb_positions
        self.hazard = hazard
        self.hazard_now = hazard_now
        self.hazard_soon = hazard_soon
        self.step = step


class Agent:
    team_id = "AlphaSearchTop1"

    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4

    # Official engine bug: action names are misleading.
    # 1 moves row -1, 2 moves row +1, 3 moves col -1, 4 moves col +1.
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    MOVE_ACTIONS = (1, 2, 3, 4)
    INF = 10**6

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self._last_signature = None
        self._bomb_radii = {}

    def act(self, obs):
        try:
            if self._is_new_game(obs):
                self.step_count = 0
                self._bomb_radii = {}
            self.step_count += 1

            state = self._parse_state(obs)
            if state is None:
                return 0

            if state.my_pos in state.hazard:
                escape = self._escape_action(state)
                if escape is not None:
                    return int(escape)

            bomb_value = self._bomb_value(state)
            if bomb_value >= self._bomb_threshold(state):
                return 5

            item_action = self._item_action(state)
            if item_action is not None:
                return int(item_action)

            farm_action = self._farm_action(state)
            if farm_action is not None:
                return int(farm_action)

            pressure_action = self._pressure_action(state)
            if pressure_action is not None:
                return int(pressure_action)

            return int(self._safe_idle_action(state))
        except Exception:
            return 0
        finally:
            try:
                self._last_signature = self._state_signature(obs)
            except Exception:
                self._last_signature = None

    def _parse_state(self, obs):
        players = obs["players"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return None

        grid = obs["map"]
        bombs = obs["bombs"]
        self._sync_bomb_memory(bombs, players)
        my = players[self.agent_id]
        my_pos = (int(my[0]), int(my[1]))
        my_bombs_left = int(my[3])
        my_radius = max(1, int(my[4]) + 1)
        enemies = []
        for idx, player in enumerate(players):
            if idx != self.agent_id and int(player[2]) == 1:
                enemies.append((int(player[0]), int(player[1]), idx))

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        hazard = self._build_hazard(grid, bombs, players)
        hazard_now = {pos for pos, timer in hazard.items() if timer <= 1}
        hazard_soon = {pos for pos, timer in hazard.items() if timer <= 3}
        return ParsedState(
            grid,
            players,
            bombs,
            my_pos,
            my_bombs_left,
            my_radius,
            enemies,
            bomb_positions,
            hazard,
            hazard_now,
            hazard_soon,
            self.step_count,
        )

    def _state_signature(self, obs):
        players = obs.get("players", [])
        bombs = obs.get("bombs", [])
        positions = tuple((int(p[0]), int(p[1]), int(p[2])) for p in players)
        return positions, len(bombs)

    def _is_new_game(self, obs):
        try:
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]
            if len(bombs) != 0 or len(players) < 4:
                return False
            h, w = grid.shape
            expected = ((1, 1), (h - 2, w - 2), (1, w - 2), (h - 2, 1))
            for idx, pos in enumerate(expected):
                if int(players[idx][2]) != 1:
                    return False
                if (int(players[idx][0]), int(players[idx][1])) != pos:
                    return False
            signature = self._state_signature(obs)
            return signature != self._last_signature
        except Exception:
            return False

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in (
            self.GRASS,
            self.ITEM_RADIUS,
            self.ITEM_CAPACITY,
        )

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES.get(int(action), (0, 0))
        return pos[0] + dx, pos[1] + dy

    def _sync_bomb_memory(self, bombs, players):
        active_keys = set()
        for bomb in bombs:
            bx, by, owner_id = int(bomb[0]), int(bomb[1]), int(bomb[3])
            key = (bx, by, owner_id)
            active_keys.add(key)
            if key not in self._bomb_radii:
                self._bomb_radii[key] = self._current_owner_radius(players, owner_id)
        for key in list(self._bomb_radii.keys()):
            if key not in active_keys:
                self._bomb_radii.pop(key, None)

    def _current_owner_radius(self, players, owner_id):
        if 0 <= owner_id < len(players):
            return max(1, int(players[owner_id][4]) + 1)
        return 1

    def _bomb_radius(self, players, owner_id, pos=None):
        if pos is not None:
            remembered = self._bomb_radii.get((int(pos[0]), int(pos[1]), int(owner_id)))
            if remembered is not None:
                return remembered
        return self._current_owner_radius(players, owner_id)

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(int(bx), int(by))}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for dist in range(1, int(radius) + 1):
                x = int(bx) + dx * dist
                y = int(by) + dy * dist
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == self.WALL:
                    break
                tiles.add((x, y))
                if cell == self.BOX:
                    break
        return tiles

    def _effective_bomb_timers(self, grid, bombs, players):
        bomb_data = []
        for bomb in bombs:
            bx, by, timer, owner_id = int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3])
            if timer <= 0:
                continue
            bomb_data.append(
                {
                    "pos": (bx, by),
                    "timer": timer,
                    "radius": self._bomb_radius(players, owner_id, pos=(bx, by)),
                }
            )

        changed = True
        while changed:
            changed = False
            for src in bomb_data:
                blast = self._blast_tiles(grid, src["pos"][0], src["pos"][1], src["radius"])
                for dst in bomb_data:
                    if src is dst:
                        continue
                    if dst["pos"] in blast and dst["timer"] > src["timer"]:
                        dst["timer"] = src["timer"]
                        changed = True
        return {entry["pos"]: entry["timer"] for entry in bomb_data}

    def _build_hazard(self, grid, bombs, players, extra_bombs=()):
        combined = []
        for bomb in bombs:
            bx, by, timer, owner_id = int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3])
            if timer > 0:
                combined.append([bx, by, timer, owner_id, self._bomb_radius(players, owner_id, pos=(bx, by))])
        for bx, by, timer, owner_id, radius in extra_bombs:
            if timer > 0:
                combined.append([int(bx), int(by), int(timer), int(owner_id), int(radius)])

        changed = True
        while changed:
            changed = False
            for src in combined:
                blast = self._blast_tiles(grid, src[0], src[1], src[4])
                for dst in combined:
                    if src is dst:
                        continue
                    if (dst[0], dst[1]) in blast and dst[2] > src[2]:
                        dst[2] = src[2]
                        changed = True

        hazard = {}
        for bx, by, timer, _owner_id, radius in combined:
            for tile in self._blast_tiles(grid, bx, by, radius):
                if timer < hazard.get(tile, self.INF):
                    hazard[tile] = timer
        return hazard

    def _hazard_timer(self, state, pos):
        return state.hazard.get(pos, self.INF)

    def _can_enter(self, grid, pos, bomb_positions, current_pos=None):
        if not self._passable(grid, pos[0], pos[1]):
            return False
        if pos in bomb_positions and pos != current_pos:
            return False
        return True

    def _safe_after_arrival(self, hazard, pos, arrival_time, buffer_steps=0):
        return hazard.get(pos, self.INF) > arrival_time + buffer_steps

    def _temporal_route_to_safe(
        self,
        state,
        hazard=None,
        bomb_positions=None,
        max_depth=12,
        require_no_hazard=True,
    ):
        hazard = state.hazard if hazard is None else hazard
        bomb_positions = state.bomb_positions if bomb_positions is None else bomb_positions
        queue = deque([(state.my_pos, 0, None)])
        seen = {(state.my_pos, 0)}

        while queue:
            pos, depth, first_action = queue.popleft()
            if depth > 0:
                no_future_blast = pos not in hazard
                safe_with_margin = self._safe_after_arrival(hazard, pos, depth, buffer_steps=2)
                if (no_future_blast if require_no_hazard else safe_with_margin):
                    return first_action
            if depth >= max_depth:
                continue

            for action in (1, 2, 3, 4, 0):
                npos = self._next_pos(pos, action)
                next_depth = depth + 1
                if action != 0 and not self._can_enter(state.grid, npos, bomb_positions, current_pos=pos):
                    continue
                if not self._safe_after_arrival(hazard, npos, next_depth, buffer_steps=0):
                    continue
                key = (npos, next_depth)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((npos, next_depth, action if first_action is None else first_action))
        return None

    def _escape_action(self, state):
        action = self._temporal_route_to_safe(state, max_depth=14, require_no_hazard=True)
        if action is not None:
            return action

        best_action = None
        best_score = -self.INF
        for action in self._valid_movement_actions(state, include_stop=False):
            npos = self._next_pos(state.my_pos, action)
            if npos in state.hazard_now:
                continue
            score = self._local_space_score(state, npos)
            score += min(8, state.hazard.get(npos, 8)) * 20
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _valid_movement_actions(self, state, include_stop=True):
        actions = [0] if include_stop else []
        for action in self.MOVE_ACTIONS:
            npos = self._next_pos(state.my_pos, action)
            if self._can_enter(state.grid, npos, state.bomb_positions, current_pos=state.my_pos):
                actions.append(action)
        return actions

    def _line_clear(self, grid, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if int(grid[ax, y]) in (self.WALL, self.BOX):
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if int(grid[x, ay]) in (self.WALL, self.BOX):
                    return False
            return True
        return False

    def _count_boxes_in_blast(self, grid, pos, radius):
        return sum(1 for tile in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[tile]) == self.BOX)

    def _count_items_in_blast(self, grid, pos, radius):
        return sum(
            1
            for tile in self._blast_tiles(grid, pos[0], pos[1], radius)
            if int(grid[tile]) in (self.ITEM_RADIUS, self.ITEM_CAPACITY)
        )

    def _enemy_escape_count_after_bomb(self, state, enemy_pos, blast):
        count = 0
        for action in self.MOVE_ACTIONS:
            npos = self._next_pos(enemy_pos, action)
            if not self._passable(state.grid, npos[0], npos[1]):
                continue
            if npos in state.bomb_positions:
                continue
            if npos not in blast and self._safe_after_arrival(state.hazard, npos, 1, buffer_steps=0):
                count += 1
        return count

    def _can_escape_after_placing(self, state):
        extra = ((state.my_pos[0], state.my_pos[1], 6, self.agent_id, state.my_radius),)
        hazard = self._build_hazard(state.grid, state.bombs, state.players, extra_bombs=extra)
        bomb_positions = set(state.bomb_positions)
        bomb_positions.add(state.my_pos)
        for ex, ey, _enemy_id in state.enemies:
            bomb_positions.add((ex, ey))
        return (
            self._temporal_route_to_safe(
                state,
                hazard=hazard,
                bomb_positions=bomb_positions,
                max_depth=12,
                require_no_hazard=True,
            )
            is not None
        )

    def _bomb_value(self, state):
        if state.my_bombs_left <= 0 or state.my_pos in state.bomb_positions:
            return -self.INF
        if state.my_pos in state.hazard:
            return -self.INF
        if not self._can_escape_after_placing(state):
            return -self.INF

        blast = self._blast_tiles(state.grid, state.my_pos[0], state.my_pos[1], state.my_radius)
        boxes = sum(1 for tile in blast if int(state.grid[tile]) == self.BOX)
        items_destroyed = sum(1 for tile in blast if int(state.grid[tile]) in (self.ITEM_RADIUS, self.ITEM_CAPACITY))

        value = 0
        direct_hits = 0
        trapped_hits = 0
        for ex, ey, _enemy_id in state.enemies:
            enemy_pos = (ex, ey)
            if enemy_pos in blast and self._line_clear(state.grid, state.my_pos, enemy_pos):
                direct_hits += 1
                exits = self._enemy_escape_count_after_bomb(state, enemy_pos, blast)
                if exits <= 1:
                    trapped_hits += 1

        value += direct_hits * 1200
        value += trapped_hits * 450
        value += boxes * (170 if state.step < 400 else 240)
        value -= items_destroyed * 180

        if state.step > 430 and boxes == 0 and direct_hits == 0:
            value += 40
        if state.my_radius <= 2 and boxes >= 1:
            value += 90
        return value

    def _bomb_threshold(self, state):
        if state.step > 430:
            return 130
        if state.step > 350:
            return 180
        return 230

    def _item_action(self, state):
        target_scores = {}
        for x in range(state.grid.shape[0]):
            for y in range(state.grid.shape[1]):
                cell = int(state.grid[x, y])
                if cell == self.ITEM_CAPACITY:
                    target_scores[(x, y)] = 520 if state.my_bombs_left <= 1 else 340
                elif cell == self.ITEM_RADIUS:
                    target_scores[(x, y)] = 500 if state.my_radius < 3 else 300
        if not target_scores:
            return None
        return self._route_to_scored_targets(state, target_scores, max_depth=24)

    def _farm_action(self, state):
        target_scores = {}
        for x in range(1, state.grid.shape[0] - 1):
            for y in range(1, state.grid.shape[1] - 1):
                pos = (x, y)
                if not self._passable(state.grid, x, y):
                    continue
                if pos in state.bomb_positions:
                    continue
                boxes = self._count_boxes_in_blast(state.grid, pos, state.my_radius)
                if boxes <= 0:
                    continue
                score = boxes * (180 if state.step > 360 else 120)
                if self._hazard_timer(state, pos) <= 3:
                    score -= 250
                target_scores[pos] = score
        if not target_scores:
            return None

        if state.my_pos in target_scores and self._bomb_value(state) >= min(120, self._bomb_threshold(state)):
            return 5
        return self._route_to_scored_targets(state, target_scores, max_depth=24)

    def _pressure_action(self, state):
        if not state.enemies:
            return None

        target_scores = {}
        for ex, ey, _enemy_id in state.enemies:
            enemy_pos = (ex, ey)
            for action in self.MOVE_ACTIONS + (0,):
                pos = self._next_pos(enemy_pos, action)
                if not self._passable(state.grid, pos[0], pos[1]):
                    continue
                if pos in state.bomb_positions:
                    continue
                dist = abs(pos[0] - state.my_pos[0]) + abs(pos[1] - state.my_pos[1])
                open_exits = self._local_space_score(state, enemy_pos)
                score = 180 - dist * 8
                if open_exits <= 20:
                    score += 120
                if state.step > 330:
                    score += 80
                target_scores[pos] = max(target_scores.get(pos, -self.INF), score)
        return self._route_to_scored_targets(state, target_scores, max_depth=18)

    def _route_to_scored_targets(self, state, target_scores, max_depth=24):
        queue = deque([(state.my_pos, 0, None)])
        seen = {state.my_pos}
        best_action = None
        best_score = -self.INF

        while queue:
            pos, depth, first_action = queue.popleft()
            if depth > max_depth:
                continue
            if pos in target_scores and first_action is not None:
                score = target_scores[pos] - depth * 18 + self._local_space_score(state, pos)
                if self._hazard_timer(state, pos) <= depth + 1:
                    score -= 1000
                if score > best_score:
                    best_score = score
                    best_action = first_action

            if depth >= max_depth:
                continue
            for action in self.MOVE_ACTIONS:
                npos = self._next_pos(pos, action)
                if npos in seen:
                    continue
                if not self._can_enter(state.grid, npos, state.bomb_positions, current_pos=pos):
                    continue
                next_depth = depth + 1
                if not self._safe_after_arrival(state.hazard, npos, next_depth, buffer_steps=1):
                    continue
                seen.add(npos)
                queue.append((npos, next_depth, action if first_action is None else first_action))
        return best_action

    def _local_space_score(self, state, pos):
        score = 0
        for action in self.MOVE_ACTIONS:
            npos = self._next_pos(pos, action)
            if self._passable(state.grid, npos[0], npos[1]) and npos not in state.bomb_positions:
                score += 12
        if int(state.grid[pos]) == self.ITEM_CAPACITY:
            score += 70
        elif int(state.grid[pos]) == self.ITEM_RADIUS:
            score += 60
        if pos in state.hazard_soon:
            score -= 80
        return score

    def _safe_idle_action(self, state):
        best_action = 0
        best_score = -self.INF
        for action in self._valid_movement_actions(state, include_stop=True):
            npos = self._next_pos(state.my_pos, action)
            if not self._safe_after_arrival(state.hazard, npos, 1, buffer_steps=1):
                continue
            score = self._local_space_score(state, npos)
            if state.enemies:
                nearest = min(abs(npos[0] - ex) + abs(npos[1] - ey) for ex, ey, _ in state.enemies)
                if nearest <= 2:
                    score -= (3 - nearest) * 30
                elif nearest <= 5 and state.step > 330:
                    score += 20
            if action == 0:
                score -= 8
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
