import numpy as np

class Features:

    def __init__(self, grid):
        self.holes = 0
        self.rows_with_holes = np.zeros(grid.shape[0])
        self.wells = 0

        for col in range(grid.shape[1]):
            isFilled = False
            well_depth = 0
            for row in range(grid.shape[0]):
                # update that filled cell exist
                if grid[row, col] == 1:
                    isFilled = True
                
                # all cells under filled cell counted as hole
                if grid[row, col] == 0 and isFilled:
                    self.holes += 1
                    self.rows_with_holes[row] = 1
                
                if isFilled:
                    # only consider wells if open from above
                    continue

                # check if it is a well by looking at left neighbor and right neighbor
                checkLeftFilled = col == 0 or grid[row, col - 1] == 1
                checkRightFilled = col == grid.shape[1] - 1 or grid[row, col + 1] == 1
                if grid[row, col] == 0 and checkLeftFilled and checkRightFilled:
                    well_depth += 1
                else:
                    # no more well, add current depth and reset
                    for x in range(1, well_depth + 1):
                        # add depth by 1 + 2 + ... + depth
                        self.wells += x
                    well_depth = 0
            
            # reach the bottom, add remaining well_depth
            for x in range(1, well_depth + 1):
                self.wells += x
            
            

    def num_holes(self):
        return self.holes
            
    def num_rows_with_holes(self):
        return np.sum(self.rows_with_holes).astype(int)
    
    def cumulative_wells(self):
        return self.wells


