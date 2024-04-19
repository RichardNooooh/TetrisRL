from nes_py.wrappers import JoypadSpace
import gym_tetris
from tetrispiece import TetrisPiece, CurrentPiece, SimulatedPiece, TETRIS_NUM_ORIENTATIONS
from tetrisgrid import TetrisGrid, CurrentGrid, SimulatedGrid
from queue import Queue

class NESTetrisEnvInterface():
    def __init__(self, tetris_type, movement_type):
        self.env = JoypadSpace(gym_tetris.make(tetris_type), movement_type)
        self.tetris_grid = CurrentGrid(self.env)
        self.tetris_piece = CurrentPiece(self.env)

    def getState(self):
        grid = self.tetris_grid.get_grid()

        self.tetris_piece.update()
        piece_tile_positions = self.tetris_piece.getAbsoluteTilePositions()
        piece_position, piece_type = self.tetris_piece.getBasePosition(), self.tetris_piece.getPieceType()
        return (grid, piece_tile_positions, (piece_position, piece_type))

    # ********************************
    # * gym_tetris Interface Methods *
    # ********************************
    
    def reset(self):
        self.env.reset() 
        return self.getState()

    def step(self, action):
        # gym_tetris's state variable is just the NES screen - not useful for features except for Deep RL agents
        _, reward, done, info = self.env.step(action)
        return self.getState(), reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    

# class SearchingUtil:

#     # ***********************************
#     # * Obtains Potential Future States *
#     # ***********************************
    
#     # adapted from MeatFighter's BFS logic

#     @staticmethod
#     def addChildToQueue(q, mark, grid, simulated_piece, new_position, new_orientation):
#         next_simulated_piece = SimulatedPiece(new_position, simulated_piece.type, new_orientation)
        
#         # check if tile is within grid bounds
#         next_tile_positions = next_simulated_piece.getAbsoluteTilePositions()
#         for next_x, next_y in next_tile_positions:
#             if next_x < 0 or next_x >= len(grid[0]) or next_y >= len(grid): # TODO check
#                 return False
            
        


#         pass

#     @staticmethod
#     def lockTetrimino():
#         pass

#     global_mark = 1 # class variable
#     @staticmethod
#     def searchLandingPositions(grid, piece_position, piece_type):
#         simulated_piece = SimulatedPiece(piece_position, piece_type)
        
#         max_rotation = TETRIS_NUM_ORIENTATIONS[piece_type] - 1
#         mark = NESTetrisEnvInterface.global_mark
#         NESTetrisEnvInterface.global_mark += 1

#         q = Queue()
#         if not SearchingUtil.addChildToQueue(q, mark, grid, simulated_piece, 
#                             simulated_piece.position, simulated_piece.orientation):
#             return False
        

        
#         while not q.empty():
#             simulated_piece = q.get()

#             # search left and right rotations
#             if max_rotation != 0:
#                 SearchingUtil.addChildToQueue(q, mark, grid, simulated_piece, simulated_piece.position,
#                         max_rotation if simulated_piece.orientation == 0 else simulated_piece.orientation - 1)
#                 if max_rotation != 1:
#                     SearchingUtil.addChildToQueue(q, grid, simulated_piece, mark,
#                         0 if simulated_piece.orientation == max_rotation else simulated_piece.orientation + 1)
            
#             # left and right shifts
#             x, y = simulated_piece.position
#             SearchingUtil.addChildToQueue(q, grid, simulated_piece, mark, (x-1, y), simulated_piece.rotation)
#             SearchingUtil.addChildToQueue(q, grid, simulated_piece, mark, (x+1, y), simulated_piece.rotation)

#             # soft drop
#             if not SearchingUtil.addChildToQueue(q, grid, simulated_piece, mark, (x, y+1), simulated_piece.rotation):
#                 SearchingUtil.lockTetrimino()
 
#         return True

