# PacMan-Graph-Navigator
Solving Pac-Man using Graph Neural Networks and DQN. Features automated node extraction and Manhattan-aligned pathfinding.

# A Graph-Based DQN Agent for Pac-Man

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project investigates a **graph-based state abstraction** approach for solving the Atari *Pac-Man* environment using Deep Q-Learning (DQN).

## Motivation

Pixel-based reinforcement learning agents often suffer from poor sample efficiency in visually rich environments. In Pac-Man, many visually close states are not necessarily reachable in a short time due to maze topology.

To address this issue, this project explores whether **explicit structural knowledge** of the environment—encoded as a graph—can improve learning stability and efficiency.

---

## Method Overview

The core idea is to decouple **high-level decision-making** from **low-level control**:

- The Pac-Man maze is converted into a graph consisting of intersections and corridors.
- The reinforcement learning agent observes a low-dimensional feature vector derived from this graph.
- At each step, the DQN selects a high-level action (e.g., a target direction or node).
- A deterministic planner (e.g., BFS or A*) executes the corresponding movement in the original game environment.

This hierarchical setup simplifies the learning task and avoids directly reasoning over raw pixel space.

---

## Key Components

- **Graph State Abstraction**  
  The observation space is constructed from graph-level features rather than RGB pixels, significantly reducing dimensionality.

- **Hierarchical Control Structure**  
  - **Policy layer (DQN):** learns strategic decisions on the graph.
  - **Control layer:** handles movement execution and collision avoidance.

- **Topology-Aware Reasoning**  
  Distances and risks are evaluated based on graph connectivity instead of Euclidean distance in image space.

---

## Implementation Details

- **Environment:** Atari Pac-Man (via `gymnasium` and `ale-py`)
- **RL Algorithm:** Deep Q-Network (DQN)
- **Policy Network:** Multi-layer perceptron (MLP)
- **Custom Environment:** `PacmanGraphEnv`, compatible with the Gymnasium API

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lansyian/PacMan-Graph-Navigator
   PacMan-Graph-Navigator

   2.  **Create a virtual environment (Recommended)**
    ```bash
    conda create -n pacman python=3.10
    conda activate pacman
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    ---

## 🏃 Usage

### 1. Train the Agent
To start training the agent from scratch:
```bash
python train.py
