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
                # if isFilled:
                #     continue

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
    
    @staticmethod
    def landing_height(grid, piece_positions):
        # piece is a list of absolute tile positions
        max_height = -1
        min_height = 1000000
        for x, y in piece_positions:
            current_height = 20 - y

            # find the highest filled tile in this column
            column = grid[:, x]
            highest_grid_tile = -1
            for column_idx, tile in enumerate(column):
                if tile == 1:
                    highest_grid_tile = column_idx
                    break

            grid_tile_height = 20 - highest_grid_tile
            landing_height = current_height - grid_tile_height

            max_height = np.maximum(landing_height, max_height)
            min_height = np.minimum(landing_height, min_height)

        return (max_height + min_height) / 2

    @staticmethod
    def eroded_piece_cells(grid, piece_positions):
        unique_y_values = set()
        for x, y in piece_positions:
            unique_y_values.add(y)

        total_eroded_lines = 0
        for y in unique_y_values:
            if sum(grid[y, :]) == len(grid[0]):
                total_eroded_lines += 1
        
        return total_eroded_lines


# test from "Why Most Decisions are Easy in Tetris Paper"
# grid = np.array([
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

# piece = [(4, 0), (4, 1), (4, 2), (4, 3)]

# assert Features.num_holes(grid) == 12
# assert Features.rows_with_holes(grid) == 6
# # assert Features.cumulative_wells(grid) == 26, "Features.cumulative_wells() = " + str(Features.cumulative_wells(grid))
# assert Features.column_transitions(grid) == 20
# # assert Features.row_transitions(grid) == 56, "Features.row_transitions() = " + str(Features.row_transitions(grid))
# assert Features.hole_depth(grid) == 12
# assert Features.landing_height(grid, piece) == 10.5, "Features.landing_height() = " + str(Features.landing_height(grid, piece))
# assert Features.eroded_piece_cells(grid, piece) == 0
