from train_dqn import DQNNetwork
import torch
from tetris import TetrisEnv, TetrisBoard
from tetrisgui import TetrisGUI
import numpy as np


### Interesting behavior: it tries to clear lines at the top instead of the bottom!
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))

    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")

    agent = torch.load("./trained_models/dqn.model")
    agent.eval()

    input()

    env = TetrisEnv()
    gui = TetrisGUI()
    gui.linkGameEnv(env)

    while True:
        board, piece, _ = env.reset()
        done = False
        while not done:
            landing_positions = TetrisEnv.onePieceSearch(piece, board)
            num_group_actions = len(landing_positions)
            state_action_array = np.zeros((num_group_actions, 20, 10), dtype=np.float32)
            for landing_index in range(num_group_actions):
                tetrimino, _ = landing_positions[landing_index]
                state_action_array[landing_index, :, :] = TetrisBoard.placePieceInBoard(board.getBoard(), tetrimino)

            state_actions_tensor = torch.from_numpy(state_action_array).unsqueeze(1).to(device)

            with torch.no_grad():
                predictions = agent(state_actions_tensor)

            action_index = torch.argmax(predictions).item()
            # state_action = state_action_array[action_index, :, :]

            (board, piece, _), _, done, _ = env.group_step(landing_positions[action_index], gui)

