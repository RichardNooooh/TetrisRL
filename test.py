import random
from tetris import TetrisBoard, Tetrimino, TetrisEnv, ACTION
from tetrisgui import TetrisGUI
from feature import Features
import time

def runTetrisGame():
    env = TetrisEnv()
    gui = TetrisGUI()

    game_over = False
    (board, current_piece, next_piece) = env.reset()

    gui.linkGameEnv(env)
    gui.runOnce()
    
    while not game_over:
        if game_over:
            (board, current_piece, next_piece) = env.reset()

        landing_positions = TetrisEnv.onePieceSearch(current_piece, board)
        actions = evaluatePositions(landing_positions, board)

        # action = getNextCommand()
        # if action is None:
        #     continue

        for action in actions:
            (board, current_piece, next_piece), reward, game_over = env.step(action)
            time.sleep(0.01)
            gui.draw()
        (board, current_piece, next_piece), reward, game_over = env.step(ACTION.SOFT_DROP)          

def evaluatePositions(landing_positions, board):
    bestScore = float('-inf')
    bestActions = None
    for propose_piece, actions in landing_positions:
        features = Features(board, propose_piece)
        landing_height = -12.63 * features.landing_height()
        eroded_piece_cells = 6.60 * features.eroded_piece_cells()
        row_transitions = -9.92 * features.row_transitions()
        column_transitions = -19.77 * features.column_transitions()
        num_holes = -13.08 * features.num_holes()
        cumulative_wells = -10.49 * features.cumulative_wells()
        hole_depth = -1.61 * features.hole_depth()
        rows_with_holes = -24.04 * features.rows_with_holes()
        score = landing_height + eroded_piece_cells + row_transitions + column_transitions + num_holes + cumulative_wells + hole_depth + rows_with_holes
        if score > bestScore:
            bestScore = score
            bestActions = actions
    return bestActions


def getNextCommand():
    command_map = {
        'a': ACTION.MOVE_LEFT,
        'd': ACTION.MOVE_RIGHT,
        'k': ACTION.ROTATE_CW,
        'j': ACTION.ROTATE_CCW,
        's': ACTION.SOFT_DROP
    }

    user_input = input().strip().lower()

    if user_input in command_map:
        return command_map[user_input]
    else:
        return None

if __name__ == '__main__':
    runTetrisGame()
