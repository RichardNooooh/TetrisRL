from agent_base import BaseAgent
from tetris.gui import TetrisGUI
from tetris.env import TetrisEnv
from tetris.feature import Features
import numpy as np

class HandwrittenBCTSAgent(BaseAgent):
    def __init__(self, file_name, show_gui=True):
        super().__init__(1, file_name, show_gui)

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
    