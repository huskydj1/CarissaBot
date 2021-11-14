import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(r"C:\\Users\\xzhou\\Downloads\\stockfish_14.1_win_x64_avx2\\stockfish_14.1_win_x64_avx2.exe")

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(time=1))
print(info['score'].white())

board.push_san("e4")
board.push_san("e5")
board.push_san("f4")
board.push_san("exf4")
board.push_san("g4")
info = engine.analyse(board, chess.engine.Limit(time=1))
print(info['score'].white())
