from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import matplotlib.pyplot as plt
import numpy as np
from state import TetrisGrid, TetrisPiece

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

tetris_grid = TetrisGrid(env) 
tetris_piece = TetrisPiece(env) # current grid

done = True
for step in range(5000):
    if done:
        _ = env.reset()
    _, reward, done, info = env.step(env.action_space.sample())

    # grid = tetris_grid.get_grid()
    # tetris_piece.update()

    env.render()

env.close()
