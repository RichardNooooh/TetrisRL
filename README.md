## CS394R Final Project: Tetris

Running the `run_*.py` scripts will run the corresponding agent. 
- `run_sarsa.py` runs and also trains a new agent. 
- `run_dqn.py` reads from `trained_models/dqn.model` to run the DQN agent without training. 
- And the BCTS agent does not train.

Interesting behavior of DQN agent: only really tries to clear lines at the top.

Required python packages:
- Pytorch
- NumPy
- PyGame (if you are using the GUI)
