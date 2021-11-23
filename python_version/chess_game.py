import chess
import chess.svg

from human_player import HumanPlayer
from bot_player import BotPlayer

if __name__ == "__main__":
    board = chess.Board('rnb1k1nr/pppp1ppp/8/2b1p3/4P2q/PPP5/3P1PPP/RNBQKBNR b KQkq - 0 4')
    player1 = HumanPlayer(chess.WHITE)
    player2 = BotPlayer(chess.BLACK)
    while True:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                print("Game Over! Black Wins.")
            else:
                print("Game Over! White Wins.")
            break

        if (board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves()):
            print("Draw.")
            break

        if board.turn == chess.WHITE:
            player_move = player1.get_move(board)
        else:
            player_move = player2.get_move(board)
        board.push(player_move)

        # write_path = 'current_board.svg'
        # with open(write_path, 'w') as f:
        #     f.write(chess.svg.board(board, size=350))
        print(board)
        print(player_move)
        print(board.fen())
