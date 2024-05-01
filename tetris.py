from enum import Enum
import numpy as np
from collections import deque
from copy import deepcopy

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

ACTION = Enum('ACTION', ["ROTATE_CW", "ROTATE_CCW", "MOVE_RIGHT", "MOVE_LEFT", "SOFT_DROP"])

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
    def transform(tetrimino, board, action):
        piece_type, new_position, new_orientation = tetrimino.getState()

        match action:
            case ACTION.MOVE_LEFT:
                new_position = (new_position[0] - 1, new_position[1])
            case ACTION.MOVE_RIGHT:
                new_position = (new_position[0] + 1, new_position[1])
            case ACTION.ROTATE_CCW:
                new_orientation = (new_orientation - 1) % TETRIS_NUM_ORIENTATIONS[piece_type]
            case ACTION.ROTATE_CW:
                new_orientation = (new_orientation + 1) % TETRIS_NUM_ORIENTATIONS[piece_type]
            case ACTION.SOFT_DROP:
                new_position = (new_position[0], new_position[1] + 1)
            case _:
                raise RuntimeError("Tetrimino.transform() received an unknown action: " + str(action))
            
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
                break
            
            # completed line
            if num_tiles_in_row == self.width:
                lines_cleared += 1
            # we are at a line that nonempty and not complete, and we have completed lines below us.
            elif lines_cleared > 0:
                self.board[y + lines_cleared + 2, :] = self.board[y + 2]

        # clear the rows above the cleared board
        self.board[:(y + lines_cleared + 3), :] = 0

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
            board_copy[tile_y+2, tile_x] = 2

        print(board_copy[2:, :])

# # Example of usage
# board = TetrisBoard()
# board.board = np.array(
# [[0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,1,0,0,0],
#  [0,1,1,1,1,1,1,0,0,0],
#  [1,1,1,1,1,1,1,0,0,0],
#  [1,1,1,1,1,1,1,0,0,0],
#  [1,1,1,0,1,1,1,0,0,0],
#  [1,0,1,0,1,1,1,0,0,0],
#  [1,1,1,1,1,1,1,1,1,1],
#  [1,1,1,1,1,1,1,1,1,1],
#  [1,1,1,1,1,1,1,1,1,1],
#  [1,1,1,1,1,1,1,1,1,1]]
# )
# board.display()
# board.clearLines()
# board.display()

class BFSNode:
    def __init__(self, tetrimino, actions):
        self.tetrimino = tetrimino
        self.actions = actions

class TetrisEnv:
    def __init__(self):
        self.board = TetrisBoard()
        self.current_piece, self.next_piece = self.spawnNewPiece(), self.spawnNewPiece()
        self.game_over = False

    def getEnvState(self):
        return self.board, self.current_piece, self.next_piece

    def spawnNewPiece(self):
        piece_type = np.random.choice(('T', 'J', 'Z', 'O', 'S', 'L', 'I'))
        return Tetrimino(piece_type, orientation=TETRIS_DEFAULT_ORIENTATIONS[piece_type])
    
    def reset(self):
        self.board.reset()
        self.current_piece, self.next_piece = self.spawnNewPiece(), self.spawnNewPiece()
        self.game_over = False

        return self.board, self.current_piece, self.next_piece
    
    def display(self):
        self.board.displayWithPiece(self.current_piece)

    @staticmethod
    def onePieceSearch(start_tetrimino, board): # TODO maybe add twoPieceSearch at some point
        visited = set()
        queue = deque([BFSNode(deepcopy(start_tetrimino), [])])
        landing_positions = []

        while len(queue) > 0:
            current_node = queue.popleft()
            current_state = current_node.tetrimino

            if current_state in visited:
                continue
            visited.add(current_state)

            # Check if this position is a landing position
            if not Tetrimino.isValidMove(board, current_state.piece_type, 
                                        (current_state.position[0], current_state.position[1] + 1), 
                                        current_state.orientation):
                landing_positions.append((current_state, current_node.actions))

            # enqueue next actions
            for action in ACTION:
                new_tetrimino = deepcopy(current_node.tetrimino)  # Start with a fresh copy for each action
                new_actions = current_node.actions + [action]
                if Tetrimino.transform(new_tetrimino, board, action):
                    queue.append(BFSNode(new_tetrimino, new_actions))

        return landing_positions

    # similar to OpenAI's gym interface
    def step(self, action):
        if self.game_over:
            raise RuntimeError("TetrisEnv.step() was called without resetting the environment.")
        
        piece_transformed = Tetrimino.transform(self.current_piece, self.board, action)
        if not piece_transformed and action == ACTION.SOFT_DROP:
            self.board.placePiece(self.current_piece)
            cleared_lines = self.board.clearLines()
            # print(cleared_lines)
            
            self.current_piece = self.next_piece
            self.next_piece = self.spawnNewPiece()

            if not self.board.canPlace(self.current_piece):
                self.game_over = True # TODO reward function
                return self.getEnvState(), -1000, self.game_over
            
            return self.getEnvState(), cleared_lines, self.game_over
        
        return self.getEnvState(), 0, self.game_over
    
        # similar to OpenAI's gym interface
    def group_step(self, next_state_tetrimino):
        if self.game_over:
            raise RuntimeError("TetrisEnv.step() was called without resetting the environment.")
        
        assert next_state_tetrimino[0].piece_type == self.current_piece.piece_type

        # piece_transformed = Tetrimino.transform(self.current_piece, self.board, action)
        self.board.placePiece(next_state_tetrimino[0])
        cleared_lines = self.board.clearLines()
        # print(cleared_lines)
        
        self.current_piece = self.next_piece
        self.next_piece = self.spawnNewPiece()

        if not self.board.canPlace(self.current_piece):
            self.game_over = True # TODO reward function
            return self.getEnvState(), -1000, self.game_over
        
        return self.getEnvState(), cleared_lines, self.game_over




# Example usage:
# Assuming we have a TetrisBoard object `board` and a Tetrimino object `current_tetrimino`
# env = TetrisEnv()
# (board, current_piece, next_piece) = env.reset()

# board.board = np.array(
# [[0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,1,0,0,0],
#  [0,1,1,1,1,1,1,0,0,0],
#  [1,1,1,1,1,1,0,0,0,0],
#  [1,1,1,1,0,0,0,0,0,0],
#  [1,1,1,0,0,0,0,0,0,0],
#  [1,0,1,0,1,1,1,1,0,0],]
# )

# current_piece.piece_type = 'T'

# landing_positions = onePieceSearch(current_piece, board)

# print(current_piece)

# for landing_position in landing_positions:
#     print("Landing Position:")
#     print("Path = ", landing_position[1])
#     board.displayWithPiece(landing_position[0])

#     print()

