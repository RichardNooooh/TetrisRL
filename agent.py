import numpy as np
import pandas as pd
from tetris import TetrisEnv
from tetrisgui import TetrisGUI
from feature import Features
import time

class BaseAgent:
    def __init__(self, num_episodes, num_runs, file_name):
        self.num_episodes = num_episodes
        self.num_runs = num_runs # for averaging
        self.num_lines_cleared = np.zeros((num_runs, num_episodes))
        self.file_name = file_name

    def train(self, run_id):
        env = TetrisEnv()
        for episode in range(self.num_episodes):
            cleared_lines = self.run_episode(env)
            self.num_lines_cleared[run_id, episode] = cleared_lines

    def run_episode(self, env):
        raise NotImplementedError
    
    def record_data(self):
        df = pd.DataFrame(self.num_lines_cleared.T, columns=[f'Run_{i+1}' for i in range(self.num_runs)])
        df.to_csv(self.file_name)

class HandwrittenBCTSAgent(BaseAgent):
    def __init__(self, num_episodes, num_runs, file_name, show_gui=True):
        super().__init__(num_episodes, num_runs, file_name)
        self.show_gui = show_gui
        if show_gui:
            self.gui = TetrisGUI()

    def run_episode(self, env):
        env = TetrisEnv()
        self.gui.linkGameEnv(env)
        self.gui.runOnce()

        game_over = False
        (board, current_piece, _) = env.reset()
        
        total_lines_cleared = 0
        while not game_over:
            if game_over:
                (board, current_piece, _) = env.reset()

            landing_positions = TetrisEnv.onePieceSearch(current_piece, board)
            best_group_action_piece, _ = self.evaluatePositions(landing_positions, board)

            (board, current_piece, _), _, game_over, info = env.group_step(best_group_action)    

            self.gui.draw()
            time.sleep(0.01)

            total_lines_cleared += info["cleared_lines"]

        return total_lines_cleared

    def evaluatePositions(self, landing_positions, board):
        bestScore = float('-inf')
        bestPiece = None
        bestActions = None
        for propose_piece, _ in landing_positions:
            features = Features(board, False)
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
                bestActions = None
        return bestPiece, bestActions
    
