from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import matplotlib.pyplot as plt
import numpy as np


env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
# print(env.unwrapped._board)


class TetrisGrid:
    def __init__(self, env):
        self.env = env.unwrapped # reference to tetris-gym environment
        self.grid = np.zeros((20, 10), dtype=np.ubyte)

    def get_grid(self):
        for i in range(20):
            for j in range(10):
                self.grid[i, j] = 1 if env.unwrapped._board[i, j] != 0xEF else 0
        return self.grid

tetris_grid = TetrisGrid(env)
print(tetris_grid.get_grid())

TETRIS_TILE_POSITIONS = [
    (( -1,  0), (  0,  0), (  1,  0), (  0, -1)),  # 00: T up
    ((  0, -1), (  0,  0), (  1,  0), (  0,  1)),  # 01: T right
    (( -1,  0), (  0,  0), (  1,  0), (  0,  1)),  # 02: T down (spawn)
    ((  0, -1), ( -1,  0), (  0,  0), (  0,  1)),  # 03: T left

    ((  0, -1), (  0,  0), ( -1,  1), (  0,  1)),  # 04: J left
    (( -1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 05: J up
    ((  0, -1), (  1, -1), (  0,  0), (  0,  1)),  # 06: J right
    (( -1,  0), (  0,  0), (  1,  0), (  1,  1)),  # 07: J down (spawn)

    (( -1,  0), (  0,  0), (  0,  1), (  1,  1)),  # 08: Z horizontal (spawn) 
    ((  1, -1), (  0,  0), (  1,  0), (  0,  1)),  # 09: Z vertical

    (( -1,  0), (  0,  0), ( -1,  1), (  0,  1)),  # 0A: O (spawn)

    ((  0,  0), (  1,  0), ( -1,  1), (  0,  1)),  # 0B: S horizontal (spawn)
    ((  0, -1), (  0,  0), (  1,  0), (  1,  1)),  # 0C: S vertical

    ((  0, -1), (  0,  0), (  0,  1), (  1,  1)),  # 0D: L right
    (( -1,  0), (  0,  0), (  1,  0), ( -1,  1)),  # 0E: L down (spawn)
    (( -1, -1), (  0, -1), (  0,  0), (  0,  1)),  # 0F: L left
    ((  1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 10: L up

    ((  0, -2), (  0, -1), (  0,  0), (  0,  1)),  # 11: I vertical
    (( -2,  0), ( -1,  0), (  0,  0), (  1,  0)),  # 12: I horizontal (spawn)
]

# current tetris piece...
class TetrisPiece:
    def __init__(self, env):
        self.env = env.unwrapped
        

        pass


plt.imshow(state)
plt.show()

