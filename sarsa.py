import numpy as np
import random
import time
from tetris import TetrisBoard, Tetrimino, TetrisEnv, ACTION
from tetrisgui import TetrisGUI
from feature import Features

def feature_vector(board, propose_piece):
    features = Features(board, propose_piece, True)

    num_holes = features.num_holes()
    rows_with_holes = features.rows_with_holes()
    cumulative_wells = features.cumulative_wells()
    column_transitions = features.column_transitions()
    row_transitions = features.row_transitions()
    hole_depth = features.hole_depth()
    landing_height = features.landing_height(propose_piece)
    eroded_piece_cells = features.eroded_piece_cells(propose_piece)

    results = np.array([num_holes, rows_with_holes, cumulative_wells, column_transitions, row_transitions, hole_depth, landing_height, eroded_piece_cells])
    return results

    # heights = features.get_heights()
    # contour = features.get_height_differences()
    # max_height = max(heights)
    # return np.concatenate((heights, contour, [max_height], [num_holes], [1])) / 20

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    # gui = TetrisGUI()
    # gui.linkGameEnv(env)
    # gui.runOnce()
    # actions = list(ACTION)
    lastDrop = [0]

    def epsilon_greedy_policy(board, w, group_actions, epsilon=0.01):
        nA = len(group_actions)
        Q = [np.dot(w, feature_vector(board, group_actions[a][0])) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            choice = np.argmax(Q)
            if lastDrop[0] > 100:
                choice = 0
                lastDrop[0] = 0
            else:
                lastDrop[0] += 1
            return choice

    w = np.zeros(8)
    

    for episode in range(num_episode):
        total_cleared = 0
        print(episode)
        print(w)
        board, piece_state, _ = env.reset()
        group_actions = TetrisEnv.onePieceSearch(piece_state, board)
        done = False
        a = epsilon_greedy_policy(board, w, group_actions)
        x = feature_vector(board, group_actions[a][0])
        z = np.zeros(8)
        Q_old = 0
        while not done:
            (board, piece_state, _), r, done, _ = env.group_step(group_actions[a])
            group_actions = TetrisEnv.onePieceSearch(piece_state, board)

            if r > 0:
                total_cleared += r
            a_prime = epsilon_greedy_policy(board, w, group_actions)
            Q = np.dot(w, x)
            x_prime = feature_vector(board, group_actions[a_prime][0])
            Q_prime = np.dot(w, x_prime)

            delta = r + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            a = a_prime
            x = x_prime
            # gui.draw()
            # time.sleep(0.05)
        print("total lines: " + str(total_cleared))
    
    return w

w = SarsaLambda(TetrisEnv(), 0.99, 0.8, 0.1, 1000)