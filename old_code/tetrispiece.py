import numpy as np

TETRIS_TILE_POSITIONS = {
    ('T', 0): (( -1,  0), (  0,  0), (  1,  0), (  0, -1)),  # 00: T up
    ('T', 1): ((  0, -1), (  0,  0), (  1,  0), (  0,  1)),  # 01: T right
    ('T', 2): (( -1,  0), (  0,  0), (  1,  0), (  0,  1)),  # 02: T down (spawn)
    ('T', 3): ((  0, -1), ( -1,  0), (  0,  0), (  0,  1)),  # 03: T left

    ('J', 0): ((  0, -1), (  0,  0), ( -1,  1), (  0,  1)),  # 04: J left
    ('J', 1): (( -1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 05: J up
    ('J', 2): ((  0, -1), (  1, -1), (  0,  0), (  0,  1)),  # 06: J right
    ('J', 3): (( -1,  0), (  0,  0), (  1,  0), (  1,  1)),  # 07: J down (spawn)

    ('Z', 0): (( -1,  0), (  0,  0), (  0,  1), (  1,  1)),  # 08: Z horizontal (spawn) 
    ('Z', 1): ((  1, -1), (  0,  0), (  1,  0), (  0,  1)),  # 09: Z vertical

    ('O', 0): (( -1,  0), (  0,  0), ( -1,  1), (  0,  1)),  # 0A: O (spawn)

    ('S', 0): ((  0,  0), (  1,  0), ( -1,  1), (  0,  1)),  # 0B: S horizontal (spawn)
    ('S', 1): ((  0, -1), (  0,  0), (  1,  0), (  1,  1)),  # 0C: S vertical

    ('L', 0): ((  0, -1), (  0,  0), (  0,  1), (  1,  1)),  # 0D: L right
    ('L', 1): (( -1,  0), (  0,  0), (  1,  0), ( -1,  1)),  # 0E: L down (spawn)
    ('L', 2): (( -1, -1), (  0, -1), (  0,  0), (  0,  1)),  # 0F: L left
    ('L', 3): ((  1, -1), ( -1,  0), (  0,  0), (  1,  0)),  # 10: L up

    ('I', 0): ((  0, -2), (  0, -1), (  0,  0), (  0,  1)),  # 11: I vertical
    ('I', 1): (( -2,  0), ( -1,  0), (  0,  0), (  1,  0)),  # 12: I horizontal (spawn)
}

TETRIS_TYPE_ORIENTATION_MAP = {
    'Tu': ('T', 0),
    'Tr': ('T', 1),
    'Td': ('T', 2),
    'Tl': ('T', 3),
    'Jl': ('J', 0),
    'Ju': ('J', 1),
    'Jr': ('J', 2),
    'Jd': ('J', 3),
    'Zh': ('Z', 0),
    'Zv': ('Z', 1),
    'O': ('O', 0),
    'Sh': ('S', 0),
    'Sv': ('S', 1),
    'Lr': ('L', 0),
    'Ld': ('L', 1),
    'Ll': ('L', 2),
    'Lu': ('L', 3),
    'Iv': ('I', 0),
    'Ih': ('I', 1)
}

TETRIS_NUM_ORIENTATIONS = {
    'T': 4, 'J': 4, 'Z': 2, 'O': 1, 'S': 2, 'L': 4, 'I': 2
}

class TetrisPiece:
    def __init__(self):
        self.position = (0, 0)
        self.type = 'T'
        self.orientation = 0
    
    def getPieceType(self):
        return (self.type, self.orientation)
    
    def getBasePosition(self):
        return self.position
    
    def getRelativeTilePositions(self):
        return TETRIS_TILE_POSITIONS[(self.type, self.orientation)]

    def getAbsoluteTilePositions(self):
        def add_pos(relative_pos):
            return tuple(np.add((self.position), relative_pos))
        
        curr_tilerelpos = TETRIS_TILE_POSITIONS[(self.type, self.orientation)]
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
        self.type, self.orientation = list(TETRIS_TYPE_ORIENTATION_MAP.values())[self.env.ram[0x42]]

class SimulatedPiece(TetrisPiece):
    def __init__(self, position, type, orientation, action):
        super().__init__()
        self.position = position
        self.type = type
        self.orientation = orientation

        self.action = action

        # used to obtain the trajectory of the piece
        # self.prev = None
        # self.visited = None

    # def rotateClockWise(self):
    #     pass
    # def rotateCounterClockWise(self):
    #     pass
    # def translateLeft(self):
    #     pass
    # def translateRight(self):
    #     pass
    # def softDrop(self):
    #     pass
    


