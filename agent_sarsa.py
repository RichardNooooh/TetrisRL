from agent_base import BaseAgent
from tetris.gui import TetrisGUI
from tetris.env import TetrisEnv
import numpy as np

class LinearSarsaLambdaAgent(BaseAgent):
    def __init__(self, num_episodes, file_name, gamma, lam, alpha, epsilon, show_gui=True):
        super().__init__(num_episodes, file_name, show_gui)
        self.w = None
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

        self.starting_epsilon = 1.0
        self.epsilon = 1.0
        self.epsilon_divide = 100_000.0
        self.ending_epsilon = epsilon

    def run_episode(self):
        self.w = np.zeros(8)

        board, piece_state, _ = self.env.reset()
        group_actions = TetrisEnv.onePieceSearch(piece_state, board)
        done = False
        a = self.epsilon_greedy_policy(board, self.w, group_actions, self.epsilon)
        x = self.feature_vector(board, group_actions[a][0])
        z = np.zeros(len(self.w))
        Q_old = 0
        total_survived_actions = 0
        total_lines_cleared = 0
        while not done:
            total_survived_actions += 1
            (board, piece_state, _), r, done, info = self.env.group_step(group_actions[a], self.gui)
            group_actions = TetrisEnv.onePieceSearch(piece_state, board)

            a_prime = self.epsilon_greedy_policy(board, self.w, group_actions, self.epsilon)
            self.epsilon = np.max((self.starting_epsilon - total_survived_actions / self.epsilon_divide, self.ending_epsilon))

            Q = np.dot(self.w, x)
            x_prime = self.feature_vector(board, group_actions[a_prime][0])
            Q_prime = np.dot(self.w, x_prime)

            delta = r + self.gamma * Q_prime - Q
            z = self.gamma * self.lam * z + (1 - self.alpha * self.gamma * self.lam * np.dot(z, x)) * x
            self.w += self.alpha * (delta + Q - Q_old) * z - self.alpha * (Q - Q_old) * x
            Q_old = Q_prime
            a = a_prime
            x = x_prime

            total_lines_cleared += info["cleared_lines"]

        return total_lines_cleared, total_survived_actions
        

    def feature_vector(self, board, propose_piece):
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

    def epsilon_greedy_policy(self, board, w, group_actions, epsilon):
        nA = len(group_actions)
        Q = [np.dot(w, self.feature_vector(board, group_actions[a][0])) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)
        
    def print_weights(self):
        np.savetxt("./trained_models/sarsa_weights.txt", self.w, delimiter=",")
        print(self.w)

    def load_weights(self, new_w):
        self.w = new_w
