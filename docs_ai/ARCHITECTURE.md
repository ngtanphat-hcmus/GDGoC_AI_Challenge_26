# System Architecture: The True King (V24) & The Honored One (V20)

Our Bomberland AI agents rely on a highly deterministic **Behavior Tree** integrated with robust **A* Pathfinding** and **BFS Hazard Detection**. Instead of relying on Reinforcement Learning (which often performs poorly in strict limit-time constraints without significant hardware), we engineered an algorithm that scales efficiently down to the sub-100ms requirement.

---

## 1. The 7-Layer Priority Tree
At every tick of the game, the agent parses the `Game State` and makes exactly one movement or bombing decision based on strict hierarchical layers. If a higher layer returns an action, the lower layers are ignored.

* **Layer 1: Immediate Survival (`_temporal_escape`)**
  If a bomb is about to explode in $\leq 2$ ticks, the agent drops everything and runs. It uses a Breadth-First Search (BFS) combined with a space-time hazard map to find the closest tile that will be safe at the time of arrival.
* **Layer 2: Safe Dropping (`Predation / Tactical bombing`)**
  The agent only places a bomb if it has pre-calculated a guaranteed escape route. 
* **Layer 3: Strategic Dodging**
  If a bomb is ticking but the explosion is $\geq 3$ ticks away, the agent routes its normal pathing *around* the future explosion zone.
* **Layer 4: Predation Trap Mode**
  The agent actively looks for enemies trapped in "dead ends" (corridors with only one exit) and drops a bomb to block the exit, guaranteeing a kill.
* **Layer 5: Box Farming**
  Evaluates clusters of wooden boxes using a custom `_bomb_value()` heuristic. It prioritizes targets where a single bomb can break multiple boxes.
* **Layer 6: Pressure Enemies**
  If no boxes are optimal, the agent paths towards the closest enemy to apply map pressure.
* **Layer 6.5: Tie-Breaker Bomb Spam (The Exploit)**
  In late-game scenarios (`step > 420`), if the agent is safe and idle, it will drop bombs in safe corners just to increment its "Bombs Placed" metric, exploiting the Tie-Breaker logic.
* **Layer 7: Safe Idle**
  Move towards the center or open space to maintain map control.

---

## 2. Overcoming the 100ms Timeout Constraint
In earlier versions (V12-V15), our bot frequently crashed. The engine imposes a strict **100ms timeout per step**. A standard A* Search running 10+ times per step (for various targets) would easily exceed this.

**The Fix:**
1. **Target Slicing:** Instead of running A* on all 50 boxes on the map, we rank them using a fast Manhattan Distance heuristic, and only run the expensive A* on the Top 10 targets `sorted_targets[:10]`.
2. **Radar Clipping (`max_d` limit):** We restricted A* to only search up to a depth of `25`. If a path isn't found within 25 steps, it aborts immediately. 

---

## 3. Minimax Threat Modeling
To survive against aggressive enemies, the bot assumes all enemies might drop a bomb on their current tile if they are within a 4-tile radius. 
We append these "ghost bombs" into the hazard map. This gives our agent an extreme "Spider-Sense" that allows it to maintain distance from erratic opponents before the bomb is even dropped.

---

## 4. Addressing Engine Quirks
During development, we discovered a mismatch between the game's display and coordinate logic:
- `engine/game.py` swapped `X` and `Y` coordinates internally. 
- Simultaneous hits on a wooden box credit BOTH agents with the score.
- Simultaneous item pickups cause the item to be "Annihilated" (deleted for both).

Our pathfinding logic naturally compensated for the axis swap without altering the foundational physics engine.
