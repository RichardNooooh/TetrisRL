import numpy as np
from copy import deepcopy
from tetris import TetrisBoard, Tetrimino, ACTION
from collections import defaultdict 

class Features:
    def __init__(self, board, propose_piece, norm=False):
        if propose_piece:
            board = deepcopy(board)
            board.placePiece(propose_piece)
        self.grid = board.board
        self.norm = norm # bool

    def num_holes(self):
        holes = 0
        for col in range(self.grid.shape[1]):
            isFilled = False
            for row in range(2, self.grid.shape[0]):
                # update that filled cell exist
                if self.grid[row, col] == 1:
                    isFilled = True
                
                # all cells under filled cell counted as hole
                if self.grid[row, col] == 0 and isFilled:
                    holes += 1

        if self.norm:
            return holes / 190.0
        return holes
            
    def rows_with_holes(self):
        rows_with_holes = np.zeros(self.grid.shape[0])
        for col in range(self.grid.shape[1]):
            isFilled = False
            for row in range(2, self.grid.shape[0]):
                # update that filled cell exist
                if self.grid[row, col] == 1:
                    isFilled = True
                
                # all cells under filled cell counted as hole
                if self.grid[row, col] == 0 and isFilled:
                    rows_with_holes[row] = 1
        
        total_row_holes = np.sum(rows_with_holes)
        if self.norm:
            return total_row_holes / 20.0
        return total_row_holes
    
    def cumulative_wells(self):
        wells = 0
        for col in range(self.grid.shape[1]):
            isFilled = False
            well_depth = 0
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] == 1:
                    isFilled = True
                # only consider wells if open from above
                if isFilled:
                    continue

                checkLeftFilled = col == 0 or self.grid[row, col - 1] == 1
                checkRightFilled = col == self.grid.shape[1] - 1 or self.grid[row, col + 1] == 1
                if self.grid[row, col] == 0 and checkLeftFilled and checkRightFilled:
                    well_depth += 1
                else:
                    # no more well, add current depth and reset
                    for x in range(1, well_depth + 1):
                        # add depth by 1 + 2 + ... + depth
                        wells += x
                    well_depth = 0
            # reach the bottom, add remaining well_depth
            for x in range(1, well_depth + 1):
                wells += x
        
        if self.norm:
            return wells / (210.0 * 5) # degenerate case of every other column filled?
        return wells

    def column_transitions(self):
        trans = 0
        isFilled = False
        for col in range(self.grid.shape[1]):
            prev = 0 # top is consider empty
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] == 1:
                    isFilled = True
                if self.grid[row, col] != prev:
                    trans += 1
                    prev = self.grid[row, col]
            # bottom of the self.grid is considered filled, count extra transition if bottom is empty
            if prev == 0 and isFilled:
                trans += 1
        if self.norm:
            return trans / 100.0 # approximate degenerate case....
        return trans
    
    def row_transitions(self):
        trans = 0
        isFilled = False
        for row in range(2, self.grid.shape[0]):
            prev = 1 # left is considered filled
            for col in range(self.grid.shape[1]):
                if self.grid[row, col] == 1:
                    isFilled = True
                if self.grid[row, col] != prev:
                    trans += 1
                    prev = self.grid[row, col]
            if prev == 0 and isFilled: # right is considered filled
                trans += 1 
        
        if self.norm:
            return trans / 200.0
        return trans

    def hole_depth(self):
        depth = 0
        for col in range(self.grid.shape[1]):
            isFilled = False
            countFilled = 0
            for row in range(2, self.grid.shape[0]):
                # update that filled cell exist
                if self.grid[row, col] == 1:
                    isFilled = True
                    countFilled += 1
                
                # only add hole depth if there exist a hole below it
                if self.grid[row, col] == 0 and isFilled:
                    depth += countFilled
                    countFilled = 0 # reset it
        
        if self.norm:
            # degenerate case: bottom row empty, everything else filled, ignoring line clears
            return depth / 190.0 
        return depth
    
    def landing_height(self, propose_piece):
        positions = Tetrimino.getPositions(*propose_piece.getState())
        heights = [20 - y for x, y in positions]
        land_height = (max(heights) + min(heights)) / 2

        if self.norm:
            return land_height / 20.0
        return land_height
    
    def eroded_piece_cells(self, propose_piece):
        positions = Tetrimino.getPositions(*propose_piece.getState())
        rowContainCurrentPiece = defaultdict(int)
        for x, y in positions:
            rowContainCurrentPiece[y + 2] += 1

        lines_cleared = 0
        current_piece_cleared = 0
        for y in rowContainCurrentPiece:
            if sum(self.grid[y, :]) == len(self.grid[0]):
                lines_cleared += 1
                current_piece_cleared += rowContainCurrentPiece[y]
        
        eroded_cells = lines_cleared * current_piece_cleared
        if self.norm:
            return eroded_cells / 16.0
        return eroded_cells
    
    def get_heights(self):
        heights = np.zeros(10)
        for col in range(self.grid.shape[1]):
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] == 1:
                    heights[col] = 22 - row
                    break
        return heights
    
    def get_height_differences(self):
        difference = np.zeros(9)
        prevHeight = -1
        for col in range(self.grid.shape[1]):
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] == 1:
                    height = 22 - row
                    if prevHeight != -1:
                        difference[col - 1] = height - prevHeight
                    prevHeight = height
                    break
        return difference


# test from "Why Most Decisions are Easy in Tetris Paper"
# board = TetrisBoard()
# board.board = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
#     [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#     [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ])

# piece = Tetrimino("I", position=(4, 10) , orientation=0)

# features = Features(board, piece)

# assert features.num_holes() == 12
# assert features.rows_with_holes() == 6
# assert features.cumulative_wells() == 26
# assert features.column_transitions() == 20
# # assert features.row_transitions() == 56
# assert features.hole_depth() == 12
# assert features.landing_height() == 10.5, print(features.landing_height())
# assert features.eroded_piece_cells() == 0