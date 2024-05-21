import pygame
import sys

class TetrisGUI:
    def __init__(self, width=510, height=600):
        pygame.init()
        self.tetris_env = None # optional
        self.width = width
        self.height = height
        self.colors = {
            'background': (10, 10, 10),
            'grid': (25, 25, 25),
            'text': (255, 255, 255),
            'placed_pieces': (100, 100, 100),
            'pieces': {
                'I': (0, 255, 255),
                'J': (255, 164, 0),
                'L': (0, 0, 255),
                'O': (255, 255, 0),
                'S': (0, 255, 0),
                'T': (128, 0, 128),
                'Z': (255, 0, 0)
            }
        }
        self.cell_size = 30
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("TetrisRL")
    
    def linkGameEnv(self, tetris_env):
        self.tetris_env = tetris_env

    def drawGrid(self, grid_height, grid_width):
        for i in range(grid_height):
            for j in range(grid_width):
                pygame.draw.rect(
                    self.screen,
                    self.colors['grid'],
                    pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1)
        
    def drawPlacedTiles(self, board):
        height, width = board.height, board.width
        for y in range(height):
            for x in range(width):
                if board.board[y+2][x] == 1:
                    pygame.draw.rect(
                        self.screen,
                        self.colors['placed_pieces'],
                        pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

    def drawTetrimino(self, piece, offset_x=0, offset_y=0):
        piece_type, position, orientation = piece.getState()
        relative_positions = piece.getRelativePositions(piece_type, orientation)
        for dx, dy in relative_positions:
            adjusted_x = position[0] + dx + offset_x
            adjusted_y = position[1] + dy + offset_y
            pygame.draw.rect(
                self.screen,
                self.colors['pieces'][piece_type],
                pygame.Rect(adjusted_x * self.cell_size, adjusted_y * self.cell_size, self.cell_size, self.cell_size))

    def drawNextTetriminoAndBox(self, next_piece):
        box_x = self.tetris_env.board.width + 1
        box_y = 2
        box_size = 5 * self.cell_size

        pygame.draw.rect(
            self.screen,
            self.colors['text'],
            pygame.Rect(box_x * self.cell_size, box_y * self.cell_size, box_size, box_size),
            2)

        # Draw the next piece within the box with an adjusted starting point
        self.drawTetrimino(next_piece, box_x - 3, box_y + 2)

    def draw(self, board=None, current_piece=None, next_piece=None):
        if self.tetris_env == None:
            assert board != None and current_piece != None and next_piece != None, \
                "TetrisGUI.draw() attempted to draw without a reference to the environment."
        else:
            assert board == None and current_piece == None and next_piece == None, \
                "TetrisGUI.draw() received to many references to the environment. " + \
                "Either link the environment or supply this function with the board, " + \
                "current piece, and next piece."
            board, current_piece, next_piece = self.tetris_env.getEnvState()
            
        self.screen.fill(self.colors['background'])
        self.drawGrid(board.height, board.width)
        self.drawPlacedTiles(board)
        self.drawTetrimino(current_piece)
        self.drawNextTetriminoAndBox(next_piece)
        pygame.display.flip()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.draw()

    def runOnce(self): # ! doesn't work when the state is None
        self.update()
        pygame.time.delay(100)
