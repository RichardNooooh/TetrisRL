from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from tetrispiece import TetrisPiece, CurrentPiece, SimulatedPiece, TETRIS_NUM_ORIENTATIONS
from tetrisgrid import TetrisGrid, CurrentGrid, SimulatedGrid
from queue import Queue
from feature import Features

SIMPLE_MOVEMENT_DICT = {
    'NOOP': 0,
    'A': 1,
    'B': 2,
    'right': 3,
    'left': 4,
    'down': 5,
}

class NESTetrisEnvInterface():
    def __init__(self, tetris_type, movement_type):
        self.env = JoypadSpace(gym_tetris.make(tetris_type), movement_type)
        self.tetris_grid = CurrentGrid(self.env)
        self.tetris_piece = CurrentPiece(self.env)
        self.action_map = self.env.get_keys_to_action()

    def getState(self):
        grid = self.tetris_grid.get_grid()

        self.tetris_piece.update()
        piece_tile_positions = self.tetris_piece.getAbsoluteTilePositions()
        raw_piece_data = self.tetris_piece.position, self.tetris_piece.type, self.tetris_piece.orientation
        return (grid, piece_tile_positions, raw_piece_data)
    
    def getActions(self):
        return self.action_map

    # ********************************
    # * gym_tetris Interface Methods *
    # ********************************
    
    def reset(self):
        self.env.reset() 
        return self.getState()

    def step(self, action):
        # gym_tetris's state variable is just the NES screen - not useful for features except for Deep RL agents
        _, reward, done, info = self.env.step(action)
        self.env.step(0)
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
        next_pieces.append(SimulatedPiece((x + 1, y), piece_type, piece_orientation, SIMPLE_MOVEMENT_DICT['right']))
        next_pieces.append(SimulatedPiece((x - 1, y), piece_type, piece_orientation, SIMPLE_MOVEMENT_DICT['left']))
        next_pieces.append(SimulatedPiece((x, y - 1), piece_type, piece_orientation, SIMPLE_MOVEMENT_DICT['down']))

        max_rotations = TETRIS_NUM_ORIENTATIONS[piece_type] - 1

        # get rotations
        if max_rotations != 0:
            next_pieces.append(SimulatedPiece((x, y), piece_type, 
                                              max_rotations if piece_orientation == 0 else piece_orientation - 1, 
                                              SIMPLE_MOVEMENT_DICT['B'])) # counterclockwise
            if max_rotations != 1:
                next_pieces.append(SimulatedPiece((x, y), piece_type, 
                                                    0 if piece_orientation == max_rotations else piece_orientation + 1, 
                                                    SIMPLE_MOVEMENT_DICT['A'])) # clockwise

        # see if these positions are valid or not
        valid_piece_candidates = []
        for piece in next_pieces:
            in_grid_bounds = True
            fits_grid_tiles = True
            tile_positions = piece.getAbsoluteTilePositions()
            for tile_position in tile_positions:
                tile_x, tile_y = tile_position
                # garbage code
                if tile_x >= 0 and tile_x < len(grid[0]) and tile_y < len(grid):
                    in_grid_bounds = False
                    break

                if grid[tile_position[1], tile_position[0]] == 1:
                    fits_grid_tiles = False
                    break
            
            if in_grid_bounds and fits_grid_tiles:
                valid_piece_candidates.append(piece)
        
        return valid_piece_candidates

    # * Note: Can move these two methods out to something else...
    @staticmethod
    def evaluateGrid(grid, piece_positions): # BCTS evaluation metric
        landing_height = Features.landing_height(grid, piece_positions)
        eroded_piece_cells = Features.eroded_piece_cells(grid, piece_positions)
        row_transitions = Features.row_transitions(grid)
        col_transitions = Features.column_transitions(grid)
        holes = Features.num_holes(grid)
        cumulative_wells = Features.cumulative_wells(grid)
        hole_depth = Features.hole_depth(grid)
        rows_with_holes = Features.rows_with_holes(grid)
        return -12.63*landing_height + 6.60*eroded_piece_cells - 9.22*row_transitions \
                - 19.77*col_transitions - 13.08*holes - 10.49*cumulative_wells -1.61*hole_depth \
                -24.04*rows_with_holes
    
    @staticmethod
    def evaluateStates(grid, piece_candidates):
        if len(piece_candidates) == 0:
            return 0

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

