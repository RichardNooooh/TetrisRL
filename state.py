# from nes_py.wrappers import JoypadSpace
# import gym_tetris
# from gym_tetris.actions import MOVEMENT
# import matplotlib.pyplot as plt
import numpy as np

TETRIS_TILE_POSITIONS = [
    (( -1,  0), (  0,  0), (  1,  0), (  0, -1)),  # 00: T up
    ((  0, -1), (  0,  0), (  1,  0), (  0,  1)),  # 01: T right
    (( -1,  0), (  0,  0), (  1,  0), (  0,  1)),  # 02: T down (spawn)
    ((  0, -1), ( -1,  0), (  0,  0), (  0,  1)),  # 03: T left

    ((  0, -1), (  0,  0), ( -1,  1), (  0,  1)),  # 04: J left
    (( -1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 05: J up
    ((  0, -1), (  1, -1), (  0,  0), (  0,  1)),  # 06: J right
    (( -1,  0), (  0,  0), (  1,  0), (  1,  1)),  # 07: J down (spawn)

    (( -1,  0), (  0,  0), (  0,  1), (  1,  1)),  # 08: Z horizontal (spawn) 
    ((  1, -1), (  0,  0), (  1,  0), (  0,  1)),  # 09: Z vertical

    (( -1,  0), (  0,  0), ( -1,  1), (  0,  1)),  # 0A: O (spawn)

    ((  0,  0), (  1,  0), ( -1,  1), (  0,  1)),  # 0B: S horizontal (spawn)
    ((  0, -1), (  0,  0), (  1,  0), (  1,  1)),  # 0C: S vertical

    ((  0, -1), (  0,  0), (  0,  1), (  1,  1)),  # 0D: L right
    (( -1,  0), (  0,  0), (  1,  0), ( -1,  1)),  # 0E: L down (spawn)
    (( -1, -1), (  0, -1), (  0,  0), (  0,  1)),  # 0F: L left
    ((  1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 10: L up

    ((  0, -2), (  0, -1), (  0,  0), (  0,  1)),  # 11: I vertical
    (( -2,  0), ( -1,  0), (  0,  0), (  1,  0)),  # 12: I horizontal (spawn)
]

TETRIS_TILE_TYPES = [
    'Tu',
    'Tr',
    'Td',
    'Tl',
    'Jl',
    'Ju',
    'Jr',
    'Jd',
    'Zh',
    'Zv',
    'O',
    'Sh',
    'Sv',
    'Lr',
    'Ld',
    'Ll',
    'Lu',
    'Iv',
    'Ih'
]


class TetrisGrid:
    def __init__(self, env):
        self.env = env.unwrapped # reference to tetris-gym environment
        self.grid = np.zeros((22, 10), dtype=np.ubyte) # extra 2 rows for the hidden items

    def get_grid(self):
        for i in range(20):
            for j in range(10):
                self.grid[i+2, j] = 1 if self.env._board[i, j] != 0xEF else 0
        return self.grid[2:, :]


class TetrisPiece:
    def __init__(self, env):
        self.env = env.unwrapped
        self.update()

    def update(self):
        self.position = (self.env.ram[0x40], self.env.ram[0x41]) 
        self.type = self.env.ram[0x42]

    def getPieceType(self):
        self.update()
        return (self.type, TETRIS_TILE_TYPES[self.type])
    
    def getBasePosition(self):
        self.update()
        return self.position
    
    def getRelativeTilePositions(self):
        return TETRIS_TILE_POSITIONS[self.type]

    def getAbsoluteTilePositions(self):
        def add_pos(relative_pos):
            return tuple(np.add((self.position), relative_pos))
        
        curr_tilerelpos = TETRIS_TILE_POSITIONS[self.type]
        return (add_pos(curr_tilerelpos[0]), add_pos(curr_tilerelpos[1]), 
                    add_pos(curr_tilerelpos[2]), add_pos(curr_tilerelpos[3]))
    
    def getGridAndPiece(self, tetrisgrid):
        self.update()
        grid_copy = tetrisgrid.grid.copy()

        tile_positions = self.getAbsoluteTilePositions()
        for tile_position in tile_positions:
            tile_x, tile_y = tile_position
            grid_copy[tile_y+2, tile_x] = 1

        return grid_copy[2:, :]


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

