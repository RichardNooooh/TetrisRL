import numpy as np
import pandas as pd
from tetris import TetrisEnv
from tetrisgui import TetrisGUI
from feature import Features

class BaseAgent:
    def __init__(self, num_episodes, file_name):
        self.num_episodes = num_episodes
        # self.num_runs = num_runs # for averaging
        self.num_lines_cleared = np.zeros(num_episodes)
        self.actions_survived = np.zeros(num_episodes)
        self.file_name = file_name
        self.env = TetrisEnv()

    def train(self):
        for episode in range(self.num_episodes):
            print("Episode:", episode)
            cleared_lines, survived_actions = self.run_episode()
            print("    Lines Cleared:", cleared_lines, "\tSurvived Frames:", survived_actions)
            self.num_lines_cleared[episode] = cleared_lines
            self.actions_survived[episode] = survived_actions

    def run_episode(self):
        raise NotImplementedError
    
    def record_data(self):
        data_dict = dict()
        data_dict["Frames Survived"] = self.actions_survived
        data_dict["Num Lines Cleared"] = self.num_lines_cleared

        df = pd.DataFrame(data_dict)
        df.to_csv(self.file_name)

class HandwrittenBCTSAgent(BaseAgent):
    def __init__(self, file_name, show_gui=True):
        super().__init__(1, file_name)
        self.show_gui = show_gui
        self.gui = None
        if show_gui:
            self.gui = TetrisGUI()
            self.gui.linkGameEnv(self.env)
            self.gui.runOnce()

    def train(self):
        lines_cleared, survived_actions = self.run_episode()
        self.num_lines_cleared[0] = lines_cleared
        self.actions_survived[0] = survived_actions

    def run_episode(self):
        game_over = False
        (board, current_piece, _) = self.env.reset()
        
        total_survived_actions = 0
        total_lines_cleared = 0
        while not game_over:
            if total_survived_actions % 1000 == 0:
                print("Lines Cleared:", total_lines_cleared, "\tSurvived Actions:", total_survived_actions)

            total_survived_actions += 1

            landing_positions = TetrisEnv.onePieceSearch(current_piece, board)
            best_group_action = self.evaluatePositions(landing_positions, board)

            (board, current_piece, _), _, game_over, info = self.env.group_step(best_group_action, self.gui)    
            total_lines_cleared += info["cleared_lines"]

        return total_lines_cleared, total_survived_actions

    def evaluatePositions(self, landing_positions, board):
        bestScore = float('-inf')
        bestPiece = None
        bestActions = None
        for propose_piece, actions in landing_positions:
            features = Features(board, propose_piece)
            landing_height = -12.63 * features.landing_height(propose_piece)
            eroded_piece_cells = 6.60 * features.eroded_piece_cells(propose_piece)
            row_transitions = -9.92 * features.row_transitions()
            column_transitions = -19.77 * features.column_transitions()
            num_holes = -13.08 * features.num_holes()
            cumulative_wells = -10.49 * features.cumulative_wells()
            hole_depth = -1.61 * features.hole_depth()
            rows_with_holes = -24.04 * features.rows_with_holes()
            score = landing_height + eroded_piece_cells + row_transitions + column_transitions + num_holes + cumulative_wells + hole_depth + rows_with_holes
            if score > bestScore:
                bestScore = score
                bestPiece = propose_piece
                bestActions = actions
        return bestPiece, bestActions
    
class LinearSarsaLambdaAgent(BaseAgent):
    def __init__(self, num_episodes, file_name, gamma, lam, alpha, epsilon, show_gui=True):
        super().__init__(num_episodes, file_name)
        self.show_gui = show_gui
        self.gui = None
        if show_gui:
            self.gui = TetrisGUI()
            self.gui.linkGameEnv(self.env)
            self.gui.runOnce()

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
