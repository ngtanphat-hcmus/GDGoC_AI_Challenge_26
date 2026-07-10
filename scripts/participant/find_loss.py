import os
import random
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

def debug_losses(agent_path):
    print('Searching for a loss...')
    env = BomberEnv(max_steps=500)
    strong_baselines = ['TacticalRuleAgent', 'SmarterRuleAgent', 'GeniusRuleAgent']
    
    match_idx = 0
    while True:
        match_idx += 1
        opponents = random.choices(strong_baselines, k=3)
        agents, names = make_agents([agent_path] + opponents, seed=None)
        
        obs = env.reset()
        done = False
        step = 0
        
        history = []
        agent_died_step = -1
        
        while not done and step < 500:
            actions = []
            for j in range(4):
                try:
                    actions.append(agents[j].act(obs))
                except Exception:
                    actions.append(0)
            
            history.append((step, obs.copy(), actions))
            if len(history) > 10:
                history.pop(0)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            
            alive_now = bool(obs['players'][0][2])
            if not alive_now and agent_died_step == -1:
                agent_died_step = step
                break
                
        if agent_died_step != -1:
            print(f'\nFound a loss at match {match_idx}, step {agent_died_step}!')
            for h_step, h_obs, h_act in history[-5:]:
                print(f'--- Step {h_step} ---')
                p = h_obs['players'][0]
                print(f'Agent Pos: ({p[0]}, {p[1]}) - Action: {h_act[0]}')
                print('Bombs:')
                for b in h_obs['bombs']:
                    print(f'  Pos: ({b[0]}, {b[1]}), Timer: {b[2]}, Owner: {b[3]}')
                mx, my = int(p[0]), int(p[1])
                grid = h_obs['map']
                H, W = grid.shape
                for i in range(max(0, mx-3), min(H, mx+4)):
                    row_str = ''
                    for j in range(max(0, my-3), min(W, my+4)):
                        if i == mx and j == my:
                            row_str += 'A '
                        else:
                            c = grid[i, j]
                            if c == 1: row_str += 'W '
                            elif c == 2: row_str += 'B '
                            elif c == 3: row_str += 'R '
                            elif c == 4: row_str += 'C '
                            else:
                                has_bomb = False
                                for b in h_obs['bombs']:
                                    if b[0] == i and b[1] == j:
                                        row_str += '* '
                                        has_bomb = True
                                        break
                                if not has_bomb:
                                    row_str += '. '
                    print(row_str)
            print('--- DEATH ---')
            break

if __name__ == '__main__':
    debug_losses('agent/my_agent_v12/agent.py')
