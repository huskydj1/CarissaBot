import chess

class HumanPlayer():
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        while True:
            move = input("Enter move in SAN notation: ")
            try:
                board.parse_san(move)
            except:
                print("Invalid Move. Try Again")
                continue
            break

        return board.parse_san(move)

