import numpy as np

class Features:
            
    @staticmethod
    def num_holes(grid):
        holes = 0
        for col in range(grid.shape[1]):
            isFilled = False
            for row in range(grid.shape[0]):
                # update that filled cell exist
                if grid[row, col] == 1:
                    isFilled = True
                
                # all cells under filled cell counted as hole
                if grid[row, col] == 0 and isFilled:
                    holes += 1
        return holes
            
    @staticmethod
    def rows_with_holes(grid):
        rows_with_holes = np.zeros(grid.shape[0])
        for col in range(grid.shape[1]):
            isFilled = False
            well_depth = 0
            prev = 0
            for row in range(grid.shape[0]):
                # update that filled cell exist
                if grid[row, col] == 1:
                    isFilled = True
                
                # all cells under filled cell counted as hole
                if grid[row, col] == 0 and isFilled:
                    rows_with_holes[row] = 1
        return np.sum(rows_with_holes).astype(int)
    
    @staticmethod
    def cumulative_wells(grid):
        wells = 0
        for col in range(grid.shape[1]):
            isFilled = False
            well_depth = 0
            for row in range(grid.shape[0]):
                if grid[row, col] == 1:
                    isFilled = True
                # only consider wells if open from above
                if isFilled:
                    continue

                checkLeftFilled = col == 0 or grid[row, col - 1] == 1
                checkRightFilled = col == grid.shape[1] - 1 or grid[row, col + 1] == 1
                if grid[row, col] == 0 and checkLeftFilled and checkRightFilled:
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

    @staticmethod
    def column_transitions(grid):
        trans = 0
        for col in range(grid.shape[1]):
            prev = 0 # top is consider empty
            for row in range(grid.shape[0]):
                if grid[row, col] != prev:
                    trans += 1
                    prev = grid[row, col]
            # bottom of the grid is considered filled, count extra transition if bottom is empty
            if prev == 0:
                trans += 1
        return trans
    
    @staticmethod
    def row_transitions(grid):
        trans = 0
        for row in range(grid.shape[0]):
            prev = 1 # left is considered filled
            for col in range(grid.shape[1]):
                if grid[row, col] != prev:
                    trans += 1
                    prev = grid[row, col]
            if prev == 0: # right is considered filled
                trans += 1 
        return trans

    @staticmethod
    def hole_depth(grid):
        depth = 0
        for col in range(grid.shape[1]):
            isFilled = False
            countFilled = 0
            for row in range(grid.shape[0]):
                # update that filled cell exist
                if grid[row, col] == 1:
                    isFilled = True
                    countFilled += 1
                
                # only add hole depth if there exist a hole below it
                if grid[row, col] == 0 and isFilled:
                    depth += countFilled
                    countFilled = 0 # reset it
        return depth
