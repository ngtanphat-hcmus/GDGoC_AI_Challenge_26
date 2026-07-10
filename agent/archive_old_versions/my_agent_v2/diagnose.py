"""
Diagnostic script: Analyze WHY AlphaSearchV2 loses/dies.
Runs matches and records detailed death analysis for every non-rank-0 game.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, random, json
import numpy as np
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

def analyze_matches(agent_path, num_matches=50, max_steps=500):
    env = BomberEnv(max_steps=max_steps)
    
    results = {
        'total': 0, 'rank0': 0, 'rank1': 0, 'rank2': 0, 'rank3': 0,
        'death_causes': [],
        'tiebreak_losses': 0,
        'self_kills': 0,
        'enemy_bomb_kills': 0,
        'chain_kills': 0,
        'endgame_stat_losses': 0,
        'draw_stat_losses': 0,
    }
    
    strong_baselines = ["TacticalRuleAgent", "SmarterRuleAgent", "GeniusRuleAgent"]
    
    for match_id in range(num_matches):
        opponents = random.choices(strong_baselines, k=3)
        agent_paths = [agent_path] + opponents
        agents, names = make_agents(agent_paths, seed=None)
        
        obs = env.reset()
        done = False
        step = 0
        
        # Track stats per step
        my_alive_history = []
        my_pos_history = []
        bomb_history = []
        death_step = -1
        death_bomb_owner = -1
        my_last_bomb_step = -1
        my_stats_history = []
        
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_order = []
        
        # Track per-player stats from engine
        all_stats_at_end = None
        
        while not done and step < max_steps:
            actions = []
            for j in range(4):
                try:
                    action = agents[j].act(obs)
                except Exception:
                    action = 0
                actions.append(action)
                if j == 0 and action == 5:
                    my_last_bomb_step = step
            
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            
            alive_now = [bool(p[2]) for p in obs["players"]]
            my_alive_history.append(alive_now[0])
            my_pos_history.append((int(obs["players"][0][0]), int(obs["players"][0][1])))
            
            for j in range(4):
                if prev_alive[j] and not alive_now[j]:
                    death_order.append(j)
                    if j == 0:
                        death_step = step
                        # Check which bombs killed us
                        # Look at bombs that exploded this step
                        for b in bomb_history:
                            pass  # simplified
            
            bomb_history = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in obs["bombs"]]
            prev_alive = alive_now
        
        # Get final stats from engine
        engine_stats = [env.players[i].stats.copy() for i in range(4)]
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [j for j in range(4) if alive_final[j]]
        
        # Calculate rank
        ranks = [0] * 4
        for j in survivors:
            ranks[j] = 0
        current_rank = 1 if len(survivors) > 0 else 0
        for group in reversed(death_order):
            ranks[group] = current_rank
            current_rank += 1
        
        results['total'] += 1
        results[f'rank{ranks[0]}'] += 1
        
        # Detailed analysis for losses (rank > 0)
        if ranks[0] > 0:
            cause = {
                'match': match_id,
                'rank': ranks[0],
                'death_step': death_step,
                'survived': 0 in survivors,
                'my_stats': engine_stats[0],
                'all_stats': engine_stats,
                'survivors': survivors,
                'last_bomb_step': my_last_bomb_step,
                'total_steps': step,
            }
            
            # Classify death cause
            if 0 in survivors and len(survivors) > 1:
                # We survived but lost by tiebreak
                cause['type'] = 'TIEBREAK_LOSS'
                results['tiebreak_losses'] += 1
                # Check which stat we lost on
                my_s = engine_stats[0]
                for other_idx in survivors:
                    if other_idx == 0:
                        continue
                    os_ = engine_stats[other_idx]
                    if os_['kills'] > my_s['kills']:
                        cause['lost_on'] = 'kills'
                    elif os_['kills'] == my_s['kills'] and os_['boxes'] > my_s['boxes']:
                        cause['lost_on'] = 'boxes'
                    elif os_['kills'] == my_s['kills'] and os_['boxes'] == my_s['boxes'] and os_['items'] > my_s['items']:
                        cause['lost_on'] = 'items'
                    else:
                        cause['lost_on'] = 'bombs'
                    break
                results['draw_stat_losses'] += 1
            elif death_step > 0:
                # We died
                if my_last_bomb_step >= 0 and death_step - my_last_bomb_step <= 8:
                    cause['type'] = 'SELF_BOMB_DEATH'
                    results['self_kills'] += 1
                elif death_step <= 50:
                    cause['type'] = 'EARLY_DEATH'
                    results['enemy_bomb_kills'] += 1
                elif death_step <= 200:
                    cause['type'] = 'MID_DEATH'
                    results['enemy_bomb_kills'] += 1
                else:
                    cause['type'] = 'LATE_DEATH'
                    results['enemy_bomb_kills'] += 1
            else:
                cause['type'] = 'UNKNOWN'
            
            results['death_causes'].append(cause)
        
        r_str = f"rank={ranks[0]}"
        stats_str = f"k={engine_stats[0]['kills']} b={engine_stats[0]['boxes']} i={engine_stats[0]['items']} bm={engine_stats[0]['bombs']}"
        print(f"Match {match_id+1}/{num_matches} | {r_str} | step={step} | {stats_str}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("LOSS ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total matches: {results['total']}")
    print(f"Rank 0 (WIN):  {results['rank0']} ({results['rank0']/results['total']*100:.1f}%)")
    print(f"Rank 1:        {results['rank1']} ({results['rank1']/results['total']*100:.1f}%)")
    print(f"Rank 2:        {results['rank2']} ({results['rank2']/results['total']*100:.1f}%)")
    print(f"Rank 3 (WORST):{results['rank3']} ({results['rank3']/results['total']*100:.1f}%)")
    
    print(f"\n--- Death Cause Breakdown ---")
    cause_counts = {}
    for c in results['death_causes']:
        t = c['type']
        cause_counts[t] = cause_counts.get(t, 0) + 1
    for t, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}")
    
    print(f"\n--- Tiebreak Loss Details ---")
    tiebreak_details = [c for c in results['death_causes'] if c['type'] == 'TIEBREAK_LOSS']
    if tiebreak_details:
        lost_on_counts = {}
        for c in tiebreak_details:
            lo = c.get('lost_on', 'unknown')
            lost_on_counts[lo] = lost_on_counts.get(lo, 0) + 1
        for lo, cnt in sorted(lost_on_counts.items(), key=lambda x: -x[1]):
            print(f"  Lost on '{lo}': {cnt}")
    
    print(f"\n--- Death Step Distribution ---")
    death_steps = [c['death_step'] for c in results['death_causes'] if c['death_step'] > 0]
    if death_steps:
        bins = [(1,50,'Early 1-50'), (51,150,'Mid 51-150'), (151,300,'Late 151-300'), (301,500,'Endgame 301-500')]
        for lo, hi, label in bins:
            cnt = sum(1 for d in death_steps if lo <= d <= hi)
            if cnt > 0:
                print(f"  {label}: {cnt} deaths (steps: {[d for d in death_steps if lo <= d <= hi]})")
    
    print(f"\n--- Stats Comparison (Losses Only) ---")
    for c in results['death_causes'][:10]:
        my_s = c['my_stats']
        print(f"  Match {c['match']+1} rank={c['rank']} type={c['type']} step={c['death_step']}: "
              f"k={my_s['kills']} b={my_s['boxes']} i={my_s['items']} bm={my_s['bombs']}")

if __name__ == "__main__":
    analyze_matches("agent/my_agent_v2", num_matches=50, max_steps=500)
