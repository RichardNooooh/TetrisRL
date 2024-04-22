import random
from tetris import TetrisBoard, Tetrimino

def runTetrisGame():
    board = TetrisBoard()
    piece = spawnNewPiece()

    game_over = False
    while not game_over:
        if piece is None or not board.canPlace(piece):
            piece = spawnNewPiece()
            if not board.canPlace(piece):
                print("Game Over!")
                game_over = True
                continue

        command = getNextCommand()
        
        if command == 'left':
            piece.moveLeft(board)
        elif command == 'right':
            piece.moveRight(board)
        elif command == 'rotateCW':
            piece.rotateCW(board)
        elif command == 'rotateCCW':
            piece.rotateCCW(board)
        elif command == 'down':
            if not piece.moveDown(board):
                board.placePiece(piece)
                _ = board.clearLines()
                piece = None

        if piece is not None:
            board.displayWithPiece(piece)

def spawnNewPiece():
    piece_types = ['T', 'J', 'Z', 'O', 'S', 'L', 'I']
    piece_type = random.choice(piece_types)
    return Tetrimino(piece_type)

def getNextCommand():
    command_map = {
        'a': 'left',
        'd': 'right',
        'k': 'rotateCW',
        'j': 'rotateCCW',
        's': 'down'
    }

    user_input = input().strip().lower()

    if user_input in command_map:
        return command_map[user_input]
    else:
        return None

if __name__ == '__main__':
    runTetrisGame()
