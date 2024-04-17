from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import matplotlib.pyplot as plt
import numpy as np
from tetrispiece import TetrisPiece, CurrentPiece, SimulatedPiece
from tetrisgrid import TetrisGrid, CurrentGrid, CurrentPiece
from feature import Features

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

tetris_grid = TetrisGrid(env) 
tetris_piece = TetrisPiece(env) # current grid

# done = True
# for step in range(5000):
#     if done:
#         _ = env.reset()
#     _, reward, done, info = env.step(env.action_space.sample())

# grid = tetris_grid.get_grid()
# tetris_piece.update()
# if step == 4000:
#     print(Features.num_holes(grid))
#     print(Features.rows_with_holes(grid))
#     print(Features.cumulative_wells(grid))
#     print(Features.column_transitions(grid))
#     print(Features.row_transitions(grid))
#     print(Features.hole_depth(grid))
#     print()

    


def feature_vector(tetris_grid, tetris_piece):
    grid = tetris_grid.get_grid()
    
    tetris_piece.update()
    piece_positions = tetris_piece.getAbsoluteTilePositions()

    num_holes = Features.num_holes(grid)
    rows_with_holes = Features.rows_with_holes(grid)
    cumulative_wells = Features.cumulative_wells(grid)
    column_transitions = Features.column_transitions(grid)
    row_transitions = Features.row_transitions(grid)
    landing_height = Features.landing_height(grid, piece_positions)
    hole_depth = Features.hole_depth(grid)

    return np.array([num_holes, rows_with_holes, cumulative_wells, column_transitions, row_transitions, landing_height, hole_depth])

def SarsaLambda(
    env, # openai gym environment
    gamma: float, # discount factor
    lam: float, # decay rate
    alpha: float, # step size
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(tetris_grid, tetris_piece, done, w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, feature_vector(tetris_grid, tetris_piece)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((7))

    for episode in range(num_episode):
        _ = env.reset()
        done = False
        a = epsilon_greedy_policy(tetris_grid, tetris_piece, done, w)
        x = feature_vector(tetris_grid, tetris_piece)
        z = np.zeros(7)
        Q_old = 0
        while not done:
            _, r, done, _ = env.step(a)
            a_prime = epsilon_greedy_policy(tetris_grid, tetris_piece, done, w)
            Q = np.dot(w, x)
            x_prime = feature_vector(tetris_grid, tetris_piece)
            Q_prime = np.dot(w, x_prime)
            delta = r + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            # s = s_prime
            a = a_prime
            x = x_prime
            env.render()

        env.close()

    return w

w = SarsaLambda(env, 0.99, 0.8, 0.1, 1)
print(w)
