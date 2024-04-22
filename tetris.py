import numpy as np

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

class Tetrimino:
    def __init__(self, piece_type, position=(5, 0), orientation=0):
        self.piece_type = piece_type
        self.position = position
        self.orientation = orientation

    def getState(self):
        return self.piece_type, self.position, self.orientation

    def rotateCW(self, board):
        new_orientation = (self.orientation + 1) % TETRIS_NUM_ORIENTATIONS[self.piece_type]
        if Tetrimino.isValidMove(board, self.piece_type, self.position, new_orientation):
            self.orientation = new_orientation

    def rotateCCW(self, board):
        new_orientation = (self.orientation - 1) % TETRIS_NUM_ORIENTATIONS[self.piece_type]
        if Tetrimino.isValidMove(board, self.piece_type, self.position, new_orientation):
            self.orientation = new_orientation

    def moveLeft(self, board):
        new_position = (self.position[0] - 1, self.position[1])
        if Tetrimino.isValidMove(board, self.piece_type, new_position, self.orientation):
            self.position = new_position

    def moveRight(self, board):
        new_position = (self.position[0] + 1, self.position[1])
        if Tetrimino.isValidMove(board, self.piece_type, new_position, self.orientation):
            self.position = new_position

    def moveDown(self, board):
        new_position = (self.position[0], self.position[1] + 1)
        if Tetrimino.isValidMove(board, self.piece_type, new_position, self.orientation):
            self.position = new_position
            return True
        return False # boolean determines whether or not piece is placed in board

    @staticmethod
    def isValidMove(board, piece_type, position, orientation):
        piece_positions = Tetrimino.getPositions(piece_type, position, orientation)
        return all(board.isWithinBounds(x, y) and not board.isOccupied(x, y) 
                   for x, y in piece_positions)

    @staticmethod
    def getPositions(piece_type, position, orientation):
        offsets = TETRIS_TILE_POSITIONS[(piece_type, orientation)]
        return tuple((position[0] + dx, position[1] + dy) for dx, dy in offsets)


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
            num_tiles_in_row = np.sum(self.board[y, :])

            # if the row is empty, then the playfield above should be empty
            if num_tiles_in_row == 0:
                break
            
            # completed line
            if num_tiles_in_row == self.width:
                lines_cleared += 1
            # we are at a line that nonempty and not complete, and we have completed lines below us.
            elif lines_cleared > 0:
                self.board[y + lines_cleared, :] = self.board[y]

        # clear the rows above the cleared board
        self.board[:(y + lines_cleared + 1), :] = 0

        return lines_cleared

    def reset(self):
        self.board.fill(0)
    
    def display(self):
        print(self.board[2:, :])

    def displayWithPiece(self, piece):
        board_copy = self.board.copy()

        tile_positions = Tetrimino.getPositions(*piece.getState())
        for tile_position in tile_positions:
            tile_x, tile_y = tile_position
            board_copy[tile_y+2, tile_x] = 1

        print(board_copy[2:, :])

# Example of usage
# board = TetrisBoard()
# board.board = np.array(
# [[0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,1,0,0,0,0],
#  [0,0,0,0,1,1,1,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [1,1,0,0,0,0,0,0,0,0],
#  [1,1,0,0,0,0,0,1,0,0],
#  [1,0,0,0,0,0,0,1,0,1],
#  [1,0,1,0,0,0,1,1,1,1],
#  [1,1,1,0,1,0,1,1,1,1],
#  [1,1,1,1,1,1,1,1,1,1]]
# )
# board.display()
# board.clearLines()
# board.display()




