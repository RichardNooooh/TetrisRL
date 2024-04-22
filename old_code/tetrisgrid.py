# from nes_py.wrappers import JoypadSpace
# import gym_tetris
# from gym_tetris.actions import MOVEMENT
# import matplotlib.pyplot as plt
import numpy as np

class TetrisGrid:
    def __init__(self):
        self.grid = np.zeros((22, 10), dtype=np.ubyte) # extra 2 rows for the hidden items
    
    def get_grid(self):
        raise NotImplementedError


class CurrentGrid(TetrisGrid):
    def __init__(self, env):
        super().__init__()
        self.env = env.unwrapped # reference to tetris-gym environment
        

    def get_grid(self):
        for i in range(20):
            for j in range(10):
                self.grid[i+2, j] = 1 if self.env._board[i, j] != 0xEF else 0
        return self.grid[2:, :]

class SimulatedGrid(TetrisGrid):
    def __init__(self):
        super().__init__()

    def get_grid(self):
        return self.grid[2:, :]

    def place_piece(self, piece_positions):
        # TODO set the tile positions to 1 based on piece_positions
        # clear lines if possible
        pass



# env = gym_tetris.make('TetrisA-v3')
# env = JoypadSpace(env, MOVEMENT)

# done = True
# for step in range(48*100):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
# # print(env.unwrapped._board)

# tetris_grid = TetrisGrid(env)
# print(tetris_grid.get_grid())

# tetris_piece = TetrisPiece(env)
# print(tetris_piece.getPieceType())
# print(tetris_piece.getBasePosition())
# print(tetris_piece.getRelativeTilePositions())
# print(tetris_piece.getAbsoluteTilePositions())
# piece_and_grid = tetris_piece.getGridAndPiece(tetris_grid)
# print(piece_and_grid)

# plt.imshow(state)
# plt.show()

