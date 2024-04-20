from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from tetrispiece import TetrisPiece, CurrentPiece, SimulatedPiece, TETRIS_NUM_ORIENTATIONS
from tetrisgrid import TetrisGrid, CurrentGrid, SimulatedGrid
from queue import Queue
from feature import Features

class NESTetrisEnvInterface():
    def __init__(self, tetris_type, movement_type):
        self.env = JoypadSpace(gym_tetris.make(tetris_type), movement_type)
        self.tetris_grid = CurrentGrid(self.env)
        self.tetris_piece = CurrentPiece(self.env)

    def getState(self):
        grid = self.tetris_grid.get_grid()

        self.tetris_piece.update()
        piece_tile_positions = self.tetris_piece.getAbsoluteTilePositions()
        raw_piece_data = self.tetris_piece.position, self.tetris_piece.type, self.tetris_piece.orientation
        return (grid, piece_tile_positions, raw_piece_data)

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
    

class OneStepSearch:
    @staticmethod
    def getNextStates(grid, simulated_piece):
        x, y = simulated_piece.position
        piece_type, piece_orientation = simulated_piece.type, simulated_piece.orientation

        next_pieces = []

        # get translations
        next_pieces.append(SimulatedPiece((x + 1, y), piece_type, piece_orientation, SIMPLE_MOVEMENT[3]))
        next_pieces.append(SimulatedPiece((x - 1, y), piece_type, piece_orientation, SIMPLE_MOVEMENT[4]))
        next_pieces.append(SimulatedPiece((x, y - 1), piece_type, piece_orientation, SIMPLE_MOVEMENT[5]))

        max_rotations = TETRIS_NUM_ORIENTATIONS[piece_type] - 1

        # get rotations
        if max_rotations != 0:
            next_pieces.append(SimulatedPiece((x, y), piece_type, max_rotations if piece_orientation == 0 else piece_orientation - 1, SIMPLE_MOVEMENT[1]))
            if max_rotations != 1:
                next_pieces.append(SimulatedPiece((x, y), piece_type, 0 if piece_orientation == max_rotations else piece_orientation + 1, SIMPLE_MOVEMENT[2]))

        # see if these positions are valid or not
        valid_piece_candidates = []
        for piece in next_pieces:
            next_x, next_y = piece.position
            in_grid_bounds = next_x < 0 or next_x >= len(grid[0]) or next_y >= len(grid)
            
            fits_grid_tiles = True
            tile_positions = piece.getAbsoluteTilePositions()
            for tile_position in tile_positions:
                if grid[tile_position[1], tile_position[0]] == 1:
                    fits_grid_tiles = False
                    break
            
            if in_grid_bounds and fits_grid_tiles:
                valid_piece_candidates.append(piece)
        
        return valid_piece_candidates


    @staticmethod
    def evaluateGrid(grid, piece_positions):
        landing_height = Features.landing_height(grid, piece_positions)
        eroded_piece_cells = Features.eroded_piece_cells(grid, piece_positions)
        row_transitions = Features.row_transitions(grid)
        col_transitions = Features.column_transitions(grid)
        holes = Features.num_holes(grid)
        cumulative_wells = Features.cumulative_wells(grid)
        return -landing_height + eroded_piece_cells - row_transitions - col_transitions - (4 * holes) - cumulative_wells
    
    @staticmethod
    def evaluateStates(grid, piece_candidates):
        if len(piece_candidates) == 0:
            return SIMPLE_MOVEMENT[0]

        best_state_action = None
        best_state_action_value = float('-inf')
        for piece in piece_candidates:
            state_action_value = OneStepSearch.evaluateGrid(grid, piece.getAbsoluteTilePositions())
            if state_action_value > best_state_action_value:
                best_state_action = piece.action
                best_state_action_value = state_action_value

        return best_state_action
            
        

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

