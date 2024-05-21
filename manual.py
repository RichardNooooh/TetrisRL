import random
from tetris.env import TetrisEnv
from tetris.state import ACTION
from tetris.gui import TetrisGUI

def runTetrisGame():
    env = TetrisEnv()
    gui = TetrisGUI()
    gui.linkGameEnv(env)
    gui.runOnce()

    game_over = False
    (board, current_piece, next_piece) = env.reset()
    while not game_over:
        if game_over:
            (board, current_piece, next_piece) = env.reset()

        action = getNextCommand()
        if action is None:
            continue

        (board, current_piece, next_piece), reward, game_over, _ = env.step(action)
        gui.draw()

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
