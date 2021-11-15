import chess
import torch
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

def convert_to_bitboard_fast(fens):
    # expect fens as a list of strings
    # input: N * str
    # of the form [board] [move] [castle] [castle] [en passant] [num moves]
    # output: N * 29 * 8 * 8

    # channel 0: whose move
    # channels 1-24: pieces
    # channel 25-28: castling

    fens = list(map(lambda x: x.split(" ")[:3], fens))

    turn_color = torch.Tensor([0 if x[1] == "w" else 1 for x in fens])
    white_k = torch.Tensor(["K" in x[2] for x in fens]).float()
    white_q = torch.Tensor(["Q" in x[2] for x in fens]).float()
    black_k = torch.Tensor(["k" in x[2] for x in fens]).float()
    black_q = torch.Tensor(["q" in x[2] for x in fens]).float()

    fens = [x[0].split("/") for x in fens]

    for i in range(1, 9):
        fens = list(map(lambda y: list(map(lambda x: x.replace(str(i), "." * i), y)), fens))

    fens = np.array([[list(y) for y in x] for x in fens])
    fens_num = np.full(fens.shape, 0.0)

    pieces = ['K', 'Q', 'B', 'R', 'N', 'P', 'k', 'q', 'b', 'r', 'n', 'p']

    for it, piece in enumerate(pieces):
        #fens_num[fens == pieces] = torch.full(fens_num[fens == pieces].size(), it + 1.0)
        #mask = torch.where(fens == pieces)
        print(fens.shape)
        fens_num[fens == np.full(fens.shape, pieces)]# = it + 1.0
        #fens_num[mask[0], mask[1]] = torch.full(torch.numel(mask[0]), it + 1.0)
    fens_num = torch.Tensor(fens_num)

    fens_stack = torch.full((fens_num.size()[0], 17, 8, 8))
    for it, tensor in zip([0, 1, 2, 3, 4], [turn_color, white_k, white_q, black_k, black_q]):
        fens_stack[:, it, :, :] = torch.swapaxes(tensor.repeat((8, 8, 1)), 0, 2)

    for it, piece in enumerate(pieces):
        fens_stack[:, it + 5, :, :] = (fens_num == it + 1).float() 

    return fens_stack
    

    





