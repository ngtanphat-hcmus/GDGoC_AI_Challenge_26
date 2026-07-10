<div align="center">
  <img src="https://img.shields.io/badge/Rank-Top_9_Global-FFD700?style=for-the-badge&logo=google" alt="Rank" />
  <img src="https://img.shields.io/badge/Win_Rate-53.18%25-4CAF50?style=for-the-badge" alt="Win Rate" />
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Algorithm-A*_&_Minimax-ff69b4?style=for-the-badge" alt="Algorithm" />
</div>

<h1 align="center">💣 Bomberland - GDGoC HCMUS AI Challenge 2026</h1>
<h3 align="center">Team: Westside Fetel</h3>

<p align="center">
  <img src="Westside_Fetel_P1.jpg" alt="Top 20 Certificate Phát" width="45%" />
  <img src="Westside_Fetel_P2.jpg" alt="Top 20 Certificate Tài" width="45%" />
</p>

<p align="center">
  <b>This repository contains the source code for our AI Agent that achieved Top 9 Global in the GDGoC HCMUS AI Challenge (Bomberland). We missed the Grand Finals (Top 8) by merely 0.04 TrueSkill points!</b>
</p>

---

## 🏆 The Journey & Story
In this 4-player Battle Royale Bomberland game, agents must collect power-ups, break wooden boxes, and blow up opponents while avoiding their own death. 

Our agent started as a basic rule-based bot but quickly evolved into an extremely aggressive, frame-perfect pathfinding AI. Through rigorous iterations, we optimized our **A* Search algorithm** to calculate safe routes within the strict **<100ms** inference limit, avoiding server-side crashes (STOP actions). Our most successful variant (**V20 - The Honored One**) played over 300 matches on the global server, achieving a staggering **53.18% Win Rate** and cementing our position at **Rank #9 Global**.

We missed the Grand Finals ticket by a microscopic gap of **0.04 TrueSkill Score** (111.93 vs 111.97). This repository is a showcase of our technical problem-solving, algorithm optimization, and meta-game exploitation.

---

## 🧠 Core Technical Highlights

What makes our agent unique is its ability to bypass performance bottlenecks and exploit the engine's meta-game scoring rules.

### 1. The 7-Layer Behavior Tree (Priority Decision-making)
Instead of deep Reinforcement Learning (which is unpredictable and slow), we engineered a deterministic, layered priority tree. Every tick, the agent evaluates the grid and makes decisions based on 7 critical layers (ranging from immediate survival to strategic tie-breaker exploits).
👉 *See [docs_ai/ARCHITECTURE.md](docs_ai/ARCHITECTURE.md) for full algorithmic details.*

### 2. Overcoming the 100ms Inference Limit
A standard A* algorithm running across a large grid with multiple enemies causes timeouts, resulting in the bot freezing. We optimized our A* by:
- Implementing strict `max_d` limits (Radar clipping).
- Using Target Slicing (only calculating routes for the Top 10 highest-value coordinates).
- Eliminating overlapping searches.

### 3. Minimax Threat Modeling & Survival Instinct
The agent constantly maps the grid for potential bomb placements from enemies within a 4-tile radius, assuming the worst-case scenario. It uses a custom **Temporal Escape (BFS)** function to calculate a 3D space-time matrix, finding tiles that will be safe when the bomb detonates.

### 4. Meta-Game Exploitations (Hacking the Tie-Breaker)
The game engine handles ties at Step 500 by ranking players based on: `Kills > Boxes > Items > Bombs Placed`.
Our AI features a **Late-game Frenzy Mode**: If `step > 350`, it prioritizes breaking boxes exponentially higher. If `step > 420` and the agent is safe but idle (no reachable enemies), it deliberately drops bombs in safe corners to rack up the "Bombs Placed" metric, ensuring victory in any Tie-breaker.

---

## 📁 Repository Structure

To keep the repository clean and professional, we have isolated our best-performing models in the `agent/` directory:

- 📂 `agent/my_agent_v20/` - **The Honored One** (Our official Top 9 submission).
- 📂 `agent/my_agent_v21/` - **Speed Optimized** (Further optimized A* target-slicing).
- 📂 `agent/my_agent_v24/` - **The True King** (The final, most advanced 7-Layer variant with Tie-breaker exploitation).
- 📂 `agent/archive_old_versions/` - Earlier iterations and baselines.
- 📂 `docs_ai/` - In-depth technical explanation of algorithms and meta-strategies.

---

## 🤝 Team & Contributions

This project was a collaborative effort. We split the workload to ensure maximum efficiency in a short hackathon timeframe.

- **Hoàng Kim Phát Tài (Project Owner / Lead):** Responsible for repository structure, CI/CD, game engine integration, baseline setup, and testing infrastructure.
- **Nguyễn Tấn Phát (Core Algorithm Developer):** Responsible for designing the AI logic, implementing A* pathfinding, Temporal Escape BFS, the 7-Layer Behavior Tree, debugging server-side Timeout issues, and Meta-game Tie-breaker strategies.

*(Ghi chú: Nếu vai trò Owner/Core của 2 bạn ngược lại, bạn chỉ cần hoán đổi tên cho nhau ở dòng trên nhé!)*

---

### How to Run (Local Simulation)
1. Install dependencies: `pip install -r requirements.txt`
2. Run a match using our Top 9 agent:
```bash
python main.py --agent agent/my_agent_v20 --players 4 --render
```