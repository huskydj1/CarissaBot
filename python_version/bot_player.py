import chess
import chess.polyglot
import random
import numpy as np
import torch
import torch.nn as nn
import math
import time
from math import log10


class SE_Block(nn.Module):
    def __init__(self, filters, se_channels):
        super().__init__()
        self.filters = filters
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters, se_channels),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, 2*filters),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 2*c, 1, 1)
        w = y[:, 0:self.filters, :, :]
        b = y[:, self.filters:2*self.filters, :, :]
        w = torch.sigmoid(w)
        return x * w.expand_as(x) + b.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, filters, se_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se = SE_Block(filters, se_channels)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + x
        out = torch.relu(out)
        return out

class CarissaNet(nn.Module):
    def __init__(self, blocks=20, filters=256, se_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(29, filters, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        
        self.residual_blocks = nn.ModuleList([ResidualBlock(filters, se_channels) for _ in range(blocks)])

        self.conv2 = nn.Conv2d(filters, 32, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        for block in self.residual_blocks:
            out = block(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out.reshape((out.size()[0], -1))
        out = self.fc(out)
        return out

def pawn_advantage_to_prob(adv):
    if adv < -1000:
        return 0
    return 1/(1+10**(adv/(-4)))

def prob_to_pawn_advantage(prob):
    if prob < 1e-10:
        return -100000
    if prob > 1-1e-10:
        return 100000
    return 4*log10(prob/(1-prob))

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
        rank, file = divmod(i,8)
        layer = piece_to_layer[p.symbol()]
        boards[layer, rank, file] = 1

        for sq in board.attacks(i):
            attack_rank, attack_file = divmod(sq,8)
            boards[layer+12, attack_rank, attack_file] += 1 # could experiment with = 1 instead of += 1

    return np.array(boards)

model = CarissaNet(blocks=8, filters=64)

sdict = torch.load('../model/model_111721_newdata_8x64_30.pt', map_location='cpu')
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
        encoding = torch.Tensor(encoding)
        if torch.cuda.is_available():
            encoding = encoding.cuda()
        output = model(torch.unsqueeze(encoding, dim=0))

    return output.item()

def tree_search(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        if board.is_stalemate() or board.is_seventyfive_moves() or board.is_insufficient_material() or board.is_fivefold_repetition():
            return 0
    
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return 1e8
            else:
                return -1e8
        
        return predict_model(board)

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
    board = chess.Board('2q3k1/p2n1ppp/4pn2/1pb5/4PB2/P2N1PP1/1PrN3P/2R2QK1 w - - 4 26')
    a1 = time.time()
    print(predict_model(board))
