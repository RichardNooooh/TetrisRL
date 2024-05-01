import numpy as np
from copy import deepcopy
from tetris import TetrisBoard, Tetrimino, ACTION
from collections import defaultdict 

class Features:
    def __init__(self, board, propose_piece, action=None):
        board = deepcopy(board)
        self.placed = False
        if action: 
            piece_transformed = Tetrimino.transform(propose_piece, board, action)
            if not piece_transformed:
                self.placed = True
                board.placePiece(propose_piece)
        else:
            self.placed = True
            board.placePiece(propose_piece)
        self.grid = board.board
        self.piece = propose_piece

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
        return np.sum(rows_with_holes).astype(int)
    
    def cumulative_wells(self):
        wells = 0
        for col in range(self.grid.shape[1]):
            isFilled = False
            well_depth = 0
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] == 1:
                    isFilled = True
                # only consider wells if open from above
                # if isFilled:
                #     continue

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
        return wells

    def column_transitions(self):
        trans = 0
        for col in range(self.grid.shape[1]):
            prev = 0 # top is consider empty
            for row in range(2, self.grid.shape[0]):
                if self.grid[row, col] != prev:
                    trans += 1
                    prev = self.grid[row, col]
            # bottom of the self.grid is considered filled, count extra transition if bottom is empty
            if prev == 0:
                trans += 1
        return trans
    
    def row_transitions(self):
        trans = 0
        for row in range(2, self.grid.shape[0]):
            prev = 1 # left is considered filled
            for col in range(self.grid.shape[1]):
                if self.grid[row, col] != prev:
                    trans += 1
                    prev = self.grid[row, col]
            if prev == 0: # right is considered filled
                trans += 1 
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
        return depth
    
    def landing_height(self):
        if self.placed: 
            positions = Tetrimino.getPositions(*self.piece.getState())
            heights = [20 - y for x, y in positions]
            return (max(heights) + min(heights)) / 2
        return 0

    def eroded_piece_cells(self):
        positions = Tetrimino.getPositions(*self.piece.getState())
        rowContainCurrentPiece = defaultdict(int)
        for x, y in positions:
            rowContainCurrentPiece[y + 2] += 1

        lines_cleared = 0
        current_piece_cleared = 0
        for y in rowContainCurrentPiece:
            if sum(self.grid[y, :]) == len(self.grid[0]):
                lines_cleared += 1
                current_piece_cleared += rowContainCurrentPiece[y]
        
        return lines_cleared * current_piece_cleared
    
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
board = TetrisBoard()
board.board = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

piece = Tetrimino("I", position=(4, 10) , orientation=0)

features = Features(board, piece)

assert features.num_holes() == 12
assert features.rows_with_holes() == 6
assert features.cumulative_wells() == 26
assert features.column_transitions() == 20
# assert features.row_transitions() == 56
assert features.hole_depth() == 12
assert features.landing_height() == 10.5, print(features.landing_height())
assert features.eroded_piece_cells() == 0