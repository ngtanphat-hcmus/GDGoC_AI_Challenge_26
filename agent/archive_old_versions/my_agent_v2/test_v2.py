"""Quick smoke test for AlphaSearchV2 agent."""
import sys, time
import numpy as np
sys.path.insert(0, r'C:\Users\PT\Downloads\Claude\GDGoC-HCMUS AI CHALLENGE 2026\Bomberland-GDGoC-AI-Challenge-main')

from agent.my_agent_v2.agent import Agent

def make_obs(grid_str, px, py, bombs=None, enemies=None):
    lines = [l.strip() for l in grid_str.strip().split('\n')]
    H, W = len(lines), len(lines[0])
    grid = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            ch = lines[r][c]
            if ch == '#': grid[r,c] = 1
            elif ch == 'B': grid[r,c] = 2
            elif ch == 'r': grid[r,c] = 3
            elif ch == 'c': grid[r,c] = 4
    
    players = np.array([[px, py, 1, 1, 1]], dtype=float)
    if enemies:
        for ex, ey in enemies:
            players = np.vstack([players, [ex, ey, 1, 1, 1]])
    else:
        players = np.vstack([players, [11, 11, 1, 1, 1]])
    
    b_arr = np.array(bombs, dtype=float) if bombs else np.zeros((0, 4))
    return {"map": grid, "players": players, "bombs": b_arr}


def test_escape():
    grid = """\
#############
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#############"""
    agent = Agent(0)
    obs = make_obs(grid, 3, 4, bombs=[[3, 3, 3, 1]], enemies=[(9, 9)])
    act = agent.act(obs)
    ok = "PASS" if act != 0 else "RISKY"
    print(f"[ESCAPE] Agent(3,4) bomb(3,3)t=3 -> action={act} [{ok}]")


def test_safe_bomb():
    grid = """\
#############
#...........#
#.#.#.#.#.#.#
#..B........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#############"""
    agent = Agent(0)
    obs = make_obs(grid, 3, 2, bombs=[], enemies=[(9, 9)])
    act = agent.act(obs)
    ok = "PASS" if act == 5 else "MISS"
    print(f"[BOMB] Agent(3,2) box(3,3) -> action={act} [{ok}]")


def test_no_suicide():
    grid = """\
#############
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#############"""
    agent = Agent(0)
    # Agent at (1,1) with a bomb blocking the only wide exit
    obs = make_obs(grid, 1, 1, bombs=[[1, 2, 5, 1], [2, 1, 5, 1]], enemies=[(9, 9)])
    act = agent.act(obs)
    ok = "PASS" if act != 5 else "FAIL-SUICIDE"
    print(f"[NO-SUICIDE] Agent(1,1) exits blocked -> action={act} [{ok}]")


def test_performance():
    grid = """\
#############
#...........#
#.#.#.#.#.#.#
#.B.B.B.....#
#.#.#.#.#.#.#
#....r......#
#.#.#.#.#.#.#
#.......c...#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#############"""
    agent = Agent(0)
    obs = make_obs(grid, 5, 5, 
                   bombs=[[3,2,5,1], [7,7,3,2]],
                   enemies=[(1,1), (11,11), (1,11)])
    
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        agent.act(obs)
        times.append((time.perf_counter() - t0) * 1000)
    
    avg = sum(times) / len(times)
    mx = max(times)
    ok = "PASS" if mx < 80 else "SLOW"
    print(f"[PERF] avg={avg:.3f}ms max={mx:.3f}ms budget=80ms [{ok}]")


def test_item_collect():
    grid = """\
#############
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...r.......#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#.#.#.#.#.#.#
#...........#
#############"""
    agent = Agent(0)
    obs = make_obs(grid, 5, 2, bombs=[], enemies=[(9, 9)])
    act = agent.act(obs)
    # Item at (5,3), agent at (5,2) -> should move RIGHT (action 4)
    ok = "PASS" if act == 4 else "MISS"
    print(f"[ITEMS] Agent(5,2) item(5,3) -> action={act} [{ok}]")


if __name__ == "__main__":
    print("=" * 50)
    print("AlphaSearchV2 - Smoke Tests")
    print("=" * 50)
    test_escape()
    test_safe_bomb()
    test_no_suicide()
    test_item_collect()
    test_performance()
    print("=" * 50)
    print("All tests complete!")
