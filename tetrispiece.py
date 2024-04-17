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

TETRIS_TILE_ROTATION_TABLE = [
    
]

class TetrisPiece:
    def __init__(self):
        self.position = (0, 0)
        self.type = 0
    
    def getPieceType(self):
        return (self.type, TETRIS_TILE_TYPES[self.type])
    
    def getBasePosition(self):
        return self.position
    
    def getRelativeTilePositions(self):
        return TETRIS_TILE_POSITIONS[self.type]

    def getAbsoluteTilePositions(self):
        def add_pos(relative_pos):
            return tuple(np.add((self.position), relative_pos))
        
        curr_tilerelpos = TETRIS_TILE_POSITIONS[self.type]
        return (add_pos(curr_tilerelpos[0]), add_pos(curr_tilerelpos[1]), 
                    add_pos(curr_tilerelpos[2]), add_pos(curr_tilerelpos[3]))
    
    def getGridAndPiece(self, grid):
        grid_copy = grid.copy()

        tile_positions = self.getAbsoluteTilePositions()
        for tile_position in tile_positions:
            tile_x, tile_y = tile_position
            grid_copy[tile_y+2, tile_x] = 1

        return grid_copy[2:, :]

class CurrentPiece(TetrisPiece):
    def __init__(self, env):
        self.env = env.unwrapped
        self.update()

    def update(self):
        self.position = (self.env.ram[0x40], self.env.ram[0x41]) 
        self.type = self.env.ram[0x42]

class SimulatedPiece(TetrisPiece):
    def __init__(self, position, type):
        super().__init__()
        self.position = position
        self.type = type

    def rotateClockWise(self):
        pass
    def rotateCounterClockWise(self):
        pass
    def translateLeft(self):
        pass
    def translateRight(self):
        pass
    def incrementDown(self):
        pass


