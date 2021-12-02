import chess
import chess.polyglot
import random
import numpy as np
import torch
import math
import time

from model import CarissaNet
# from data_manip import pawn_to_prob, prob_to_pawn

piece_to_layer = {
    'R': 1,
    'N': 2,
    'B': 3,
    'Q': 4,
    'K': 5,
    'P': 6,
    'p': 7,
    'k': 8,
    'q': 9,
    'b': 10,
    'n': 11,
    'r': 12
}

def convert_to_bitboard(fen):
    boards = np.zeros((29,8,8), dtype=np.uint8)

    board = chess.Board(fen)
    turn_color = board.turn

    cr = board.castling_rights
    wkcastle = bool(cr & chess.BB_H1)
    wqcastle = bool(cr & chess.BB_A1)
    bkcastle = bool(cr & chess.BB_H8)
    bqcastle = bool(cr & chess.BB_A8)

    boards[0, :, :] = turn_color
    boards[25, :, :] = wkcastle
    boards[26, :, :] = wqcastle
    boards[27, :, :] = bkcastle
    boards[28, :, :] = bqcastle

    piece_map = board.piece_map()
    for i, p in piece_map.items():
        piece_rank, piece_file = divmod(i,8)
        layer = piece_to_layer[p.symbol()]
        boards[layer, piece_rank, piece_file] = 1

        for sq in board.attacks(i):
            attack_rank, attack_file = divmod(sq,8)
            boards[layer+12, attack_rank, attack_file] += 1 # could experiment with = 1 instead of += 1

    return np.array(boards)

model = CarissaNet(blocks=10, filters=128)

sdict = torch.load('../model/model_112021_bigdata_10x128_40.pt', map_location='cpu')
for key in list(sdict.keys()):
    if key.startswith("module."):
        sdict[key[7:]] = sdict.pop(key)

model.load_state_dict(sdict)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

def predict_model(board):
    encoding = convert_to_bitboard(board.fen())
    with torch.no_grad():
        encoding = torch.unsqueeze(torch.Tensor(encoding), dim=0)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(encoding)
    # print(encoding.size())
    return output.item()

def predict_model_batched(board):
    if board.turn == chess.WHITE:
        start = True
        stalemate = False
        for move in board.legal_moves:
            board.push(move)
            if not stalemate:
                if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
                    stalemate = True
                    board.pop()
                    continue

            if board.is_checkmate():
                if board.turn == chess.BLACK:
                    board.pop()
                    return 1e8
                else:
                    board.pop()
                    continue

            if start:
                encodings = convert_to_bitboard(board.fen())
                encodings = torch.unsqueeze(torch.Tensor(encodings), dim=0)
                start = False
            else:
                encoding = convert_to_bitboard(board.fen())
                encoding = torch.unsqueeze(torch.Tensor(encoding), dim=0)
                encodings = torch.cat((encodings, encoding), dim=0)

            board.pop()

        if start:
            if stalemate:
                return 0.5
            return -1e8

        if torch.cuda.is_available():
            encodings = encodings.cuda()
        # print(encodings.size())
        outputs = model(encodings)
        return torch.max(outputs).item()

    else:
        start = True
        stalemate = False
        for move in board.legal_moves:
            board.push(move)
            if not stalemate:
                if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
                    stalemate = True
                    board.pop()
                    continue

            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    board.pop()
                    return -1e8
                else:
                    board.pop()
                    continue

            if start:
                encodings = convert_to_bitboard(board.fen())
                encodings = torch.unsqueeze(torch.Tensor(encodings), dim=0)
                start = False
            else:
                encoding = convert_to_bitboard(board.fen())
                encoding = torch.unsqueeze(torch.Tensor(encoding), dim=0)
                encodings = torch.cat((encodings, encoding), dim=0)

            board.pop()

        if start:
            if stalemate:
                return 0.5
            return -1e8

        if torch.cuda.is_available():
            encodings = encodings.cuda()

        outputs = model(encodings)
        return torch.min(outputs).item()

def tree_search(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
            return 0.5
    
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -1e8
            else:
                return 1e8
        
        return predict_model_batched(board)

    if maximizing_player:
        value = -1e10
        for move in board.legal_moves:
            board.push(move)
            value = max(value, tree_search(board, depth-1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = 1e10
        for move in board.legal_moves:
            board.push(move)
            value = min(value, tree_search(board, depth-1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

class BotPlayer:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        opening_books = ['vitamin17','Human','Titans','baron30']

        for book in opening_books:
            with chess.polyglot.open_reader(f'data/{book}.bin') as reader:
                board_hash = chess.polyglot.zobrist_hash(board)
                if reader.get(board_hash) is not None:
                    print(f'Found in {book} book.')
                    return reader.weighted_choice(board_hash).move

        print('Computing move.')

        depth = 1
        best_value = 1e10
        for move in board.legal_moves:
            board.push(move)
            value = tree_search(board, depth, -1e10, 1e10, True)
            board.pop()
            print(move, value)

            if value < best_value:
                best_value = value
                best_move = move
                
        print(best_move, best_value)
        return best_move

if __name__ == '__main__':
    board = chess.Board('rnb1k1nr/1pp1b2p/p1q1p1p1/3NB1Q1/3P4/4P3/PPP2PPP/R3KB1R b KQkq - 0 11')
    start = True
    print(predict_model(board))

    print(time.time())
    num = 0
    total = 0
    should_break = False
    for move in board.legal_moves:
        board.push(move)
        for second_move in board.legal_moves:
            board.push(second_move)
            num += 1
            num_moves = len(list(board.legal_moves))
            total += num_moves
            print(num_moves, num)
            if num > 100:
                print(total)
                should_break = True
                break
            for third_move in board.legal_moves:
                board.push(third_move)    
                if start:
                    encodings = convert_to_bitboard(board.fen())
                    encodings = torch.unsqueeze(torch.Tensor(encodings), dim=0)
                    start = False
                else:
                    encoding = convert_to_bitboard(board.fen())
                    encoding = torch.unsqueeze(torch.Tensor(encoding), dim=0)
                    encodings = torch.cat((encodings, encoding), dim=0)
                board.pop()
            board.pop()
        board.pop()
        if should_break:
            break
    print(time.time())

    if torch.cuda.is_available():
        encodings = encodings.cuda()
    print(time.time())
    outputs = model(encodings)
    
    print(time.time())

    print(encodings.size())
    print(outputs.size())

    # print(time.time())
    # print(predict_model(board))
    # print(time.time())
    # print(predict_model(board))
    # print(time.time())
    # print(predict_model_batched(board))
    # print(time.time())
    # print(predict_model_batched(board))
    # print(time.time())
