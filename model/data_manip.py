import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10

def eval_to_pawn(eval):
    try:
        res = int(eval)
    except ValueError:
        res = 10000 if eval[1] == '+' else -10000

    res = min(res, 10000)
    res = max(res, -10000)
    return res/100

def pawn_to_prob(adv):
    if adv < -100:
        return 0
    if adv > 100:
        return 1
    
    return 1/(1+10**(adv/(-4)))

def prob_to_pawn(prob):
    if prob > 1-1e-10:
        return 1e6
    if prob < 1e-10:
        return -1e6

    return -4*log10(1/prob-1)

pieces_to_layer = {
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
    board = chess.Board(fen)

    bitboard = np.zeros((29,8,8), dtype=np.int8)
    turn_color = board.turn
    bitboard[0,:,:] = turn_color

    cr = board.castling_rights
    wkcastle = bool(cr & chess.BB_H1)
    wqcastle = bool(cr & chess.BB_A1)
    bkcastle = bool(cr & chess.BB_H8)
    bqcastle = bool(cr & chess.BB_A8)

    bitboard[25,:,:] = wkcastle
    bitboard[26,:,:] = wqcastle
    bitboard[27,:,:] = bkcastle
    bitboard[28,:,:] = bqcastle

    piece_map = board.piece_map()
    for i,p in piece_map.items():
        rank, file = divmod(i,8)
        layer = pieces_to_layer[p.symbol()]
        bitboard[layer,rank,file] = 1

        for sq in board.attacks(i):
            attack_rank, attack_file = divmod(sq,8)
            bitboard[layer+12, attack_rank, attack_file] = 1

    return bitboard