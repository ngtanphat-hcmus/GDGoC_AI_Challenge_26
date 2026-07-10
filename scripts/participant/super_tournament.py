import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import random
from collections import defaultdict

parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

def run_tournament(num_matches=200):
    agents_to_compete = [
        "agent/my_agent_v20/agent.py",
        "agent/my_agent_v18d/agent.py",
        "agent/my_agent_v16/agent.py",
        "agent/my_agent_v12/agent.py"
    ]
    
    agent_names = ["V20 (Đỉnh Phong)", "V18d (Sát Thủ)", "V16 (Bất Tử)", "V12 (Baseline)"]
    
    stats = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "steps": 0, "deaths": 0})
    
    env = BomberEnv(max_steps=500)
    
    print(f"=== BẮT ĐẦU SIÊU GIẢI ĐẤU TOURNAMENT ({num_matches} TRẬN) ===")
    print("Thành phần tham dự:")
    for i in range(4):
        print(f"- {agent_names[i]}: {agents_to_compete[i]}")
        
    for i in range(num_matches):
        if i % 10 == 0:
            print(f"Đang chạy trận {i+1}/{num_matches}...")
            
        # Shuffle positions
        indices = list(range(4))
        random.shuffle(indices)
        
        current_paths = [agents_to_compete[idx] for idx in indices]
        agents, names = make_agents(current_paths, seed=None)
        
        obs = env.reset()
        done = False
        step = 0
        
        prev_alive = [True, True, True, True]
        
        while not done and step < 500:
            actions = []
            for j in range(4):
                try:
                    action = agents[j].act(obs)
                except Exception:
                    action = 0
                actions.append(action)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            
            alive_now = [bool(p[2]) for p in obs["players"]]
            for j in range(4):
                if prev_alive[j] and not alive_now[j]:
                    original_idx = indices[j]
                    stats[original_idx]["deaths"] += 1
            prev_alive = alive_now
            
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [j for j in range(4) if alive_final[j]]
        
        for j in range(4):
            original_idx = indices[j]
            stats[original_idx]["steps"] += step
            
            if alive_final[j]:
                if len(survivors) == 1:
                    stats[original_idx]["wins"] += 1
                else:
                    stats[original_idx]["draws"] += 1
            else:
                stats[original_idx]["losses"] += 1

    print("\n=== KẾT QUẢ CHUNG CUỘC TOURNAMENT 4 HỆ HÀNG ĐẦU ===")
    print(f"{'Agent':<20} | {'Win Rate':<10} | {'Draw Rate':<10} | {'Loss Rate':<10} | {'Avg Steps':<10}")
    print("-" * 65)
    
    # Sort by wins
    sorted_indices = sorted(range(4), key=lambda x: stats[x]["wins"], reverse=True)
    
    for idx in sorted_indices:
        st = stats[idx]
        wr = (st["wins"] / num_matches) * 100
        dr = (st["draws"] / num_matches) * 100
        lr = (st["losses"] / num_matches) * 100
        ast = st["steps"] / num_matches
        name = agent_names[idx]
        print(f"{name:<20} | {wr:>8.1f}% | {dr:>8.1f}% | {lr:>8.1f}% | {ast:>8.1f}")
        
if __name__ == "__main__":
    run_tournament(200)
