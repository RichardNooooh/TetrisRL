import numpy as np
from enum import IntEnum

# positions based on NES Tetris
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

TETRIS_NUM_ORIENTATIONS = {
    'T': 4, 'J': 4, 'Z': 2, 'O': 1, 'S': 2, 'L': 4, 'I': 2
}

TETRIS_DEFAULT_ORIENTATIONS = {
    'T': 2, 'J': 3, 'Z': 0, 'O': 0, 'S': 0, 'L': 1, 'I': 1
}

# ACTION = IntEnum('ACTION', ["ROTATE_CW", "ROTATE_CCW", "MOVE_RIGHT", "MOVE_LEFT", "SOFT_DROP"])
class ACTION(IntEnum):
    SOFT_DROP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4

class Tetrimino:
    def __init__(self, piece_type, position=(5, 0), orientation=0):
        self.piece_type = piece_type
        self.position = position
        self.orientation = orientation

    # __hash__ and __eq__ used for BFS
    def __hash__(self):
        return hash((self.position, self.orientation))

    def __eq__(self, other):
        return (self.position, self.orientation) == (other.position, other.orientation)

    def getState(self):
        return self.piece_type, self.position, self.orientation

    @staticmethod
    def actionMap(action, position, orientation, piece_type):
        match action:
            case ACTION.MOVE_LEFT:
                position = (position[0] - 1, position[1])
            case ACTION.MOVE_RIGHT:
                position = (position[0] + 1, position[1])
            case ACTION.ROTATE_CCW:
                orientation = (orientation - 1) % TETRIS_NUM_ORIENTATIONS[piece_type]
            case ACTION.ROTATE_CW:
                orientation = (orientation + 1) % TETRIS_NUM_ORIENTATIONS[piece_type]
            case ACTION.SOFT_DROP:
                position = (position[0], position[1] + 1)
            case _:
                raise RuntimeError("Tetrimino.transform() received an unknown action: " + str(action))
        return position, orientation, piece_type

    @staticmethod
    def transform(tetrimino, board, action):
        piece_type, position, orientation = tetrimino.getState()

        new_position, new_orientation, piece_type = Tetrimino.actionMap(action, position, orientation, piece_type)
            
        if Tetrimino.isValidMove(board, piece_type, new_position, new_orientation):
            tetrimino.position = new_position
            tetrimino.orientation = new_orientation
            return True
        return False

    @staticmethod
    def isValidMove(board, piece_type, position, orientation):
        piece_positions = Tetrimino.getPositions(piece_type, position, orientation)
        return all(board.isWithinBounds(x, y) and not board.isOccupied(x, y) 
                   for x, y in piece_positions)

    @staticmethod
    def getPositions(piece_type, position, orientation):
        offsets = TETRIS_TILE_POSITIONS[(piece_type, orientation)]
        return tuple((position[0] + dx, position[1] + dy) for dx, dy in offsets)
    
    @staticmethod
    def getRelativePositions(piece_type, orientation):
        return TETRIS_TILE_POSITIONS[(piece_type, orientation)]


class TetrisBoard:
    def __init__(self, width=10, height=20):
        self.height = height
        self.width = width
        self.board = np.zeros((height + 2, width), dtype=int)

    def isWithinBounds(self, x, y):
        return 0 <= x < self.width and -2 <= y < self.height

    def isOccupied(self, x, y):
        return self.board[y + 2, x] == 1

    def placePiece(self, tetrimino):
        for x, y in Tetrimino.getPositions(*tetrimino.getState()):
            self.board[y + 2, x] = 1

    def canPlace(self, tetrimino):
        for x, y in Tetrimino.getPositions(*tetrimino.getState()):
            if not self.isWithinBounds(x, y) or self.isOccupied(x, y):
                return False
        return True

    def clearLines(self):
        lines_cleared = 0

        # iterate bottom up
        for y in range(self.height - 1, -1, -1):
            num_tiles_in_row = np.sum(self.board[y + 2, :])

            # if the row is empty, then the playfield above should be empty
            if num_tiles_in_row == 0:
                y += 1
                break
            
            # completed line
            if num_tiles_in_row == self.width:
                lines_cleared += 1
            # we are at a line that nonempty and not complete, and we have completed lines below us.
            elif lines_cleared > 0:
                self.board[y + lines_cleared + 2, :] = self.board[y + 2]

        # clear the rows above the cleared board
        self.board[:(y + lines_cleared + 2), :] = 0

        return lines_cleared

    def reset(self):
        self.board.fill(0)
    
    def getBoard(self):
        return self.board[2:, :].copy()

    def getBoardWithPiece(self, piece):
        board_copy = self.getBoard().astype(dtype=np.float32)

        tile_positions = Tetrimino.getPositions(*piece.getState())
        for tile_position in tile_positions:
            tile_x, tile_y = tile_position
            board_copy[tile_y, tile_x] = 0.5

        return board_copy
    
    @staticmethod
    def placePieceInBoard(board, piece):
        board = board.astype(dtype=np.float32)
        tile_positions = Tetrimino.getPositions(*piece.getState())
        for tile_position in tile_positions:
            tile_x, tile_y = tile_position
            board[tile_y, tile_x] = 0.5

        return board

