import numpy as np
from collections import deque
from copy import deepcopy
import time

from .state import Tetrimino, TetrisBoard, TETRIS_DEFAULT_ORIENTATIONS, ACTION

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
        
        # self.step_counter += 1

        info = dict()
        info["placed_piece"] = None
        info["cleared_lines"] = 0

        piece_transformed = Tetrimino.transform(self.current_piece, self.board, action)

        # after movement, if we didn't already try to soft drop, fall by 1
        # if action != ACTION.SOFT_DROP and self.step_counter % 10 == 0:
        #     piece_transformed = Tetrimino.transform(self.current_piece, self.board, ACTION.SOFT_DROP)
        #     action = ACTION.SOFT_DROP

        if not piece_transformed and action == ACTION.SOFT_DROP:
            info["placed_piece"] = self.current_piece, deepcopy(self.board)

            self.board.placePiece(self.current_piece)
            cleared_lines = self.board.clearLines()
            info["cleared_lines"] = cleared_lines
            
            self.current_piece = self.next_piece
            self.next_piece = self.spawnNewPiece()

            if not self.board.canPlace(self.current_piece):
                self.game_over = True
                return self.getEnvState(), -10000, self.game_over, info
            
            return self.getEnvState(), cleared_lines*10, self.game_over, info

        
        return self.getEnvState(), 0.001, self.game_over, info
    
    # similar to OpenAI's gym interface
    def group_step(self, next_state_tetrimino, gui=None):
        if self.game_over:
            raise RuntimeError("TetrisEnv.step() was called without resetting the environment.")
        
        if gui:
            actions = next_state_tetrimino[1]
            for action in actions:
                succeeded = Tetrimino.transform(self.current_piece, self.board, action)
                assert succeeded
                gui.draw()
                time.sleep(0.02)

        # piece_transformed = Tetrimino.transform(self.current_piece, self.board, action)

        assert next_state_tetrimino[0].piece_type == self.current_piece.piece_type
        self.board.placePiece(next_state_tetrimino[0])

        if gui: gui.draw()

        cleared_lines = self.board.clearLines()
        # print(cleared_lines)

        info = dict()
        info["cleared_lines"] = cleared_lines
        
        self.current_piece = self.next_piece
        self.next_piece = self.spawnNewPiece()

        if not self.board.canPlace(self.current_piece):
            self.game_over = True # TODO reward function
            return self.getEnvState(), -10000, self.game_over, info
        
        return self.getEnvState(), 0.01 + 10*cleared_lines, self.game_over, info
