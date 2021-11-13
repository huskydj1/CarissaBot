import chess
import torch

import data_manip

class ChessDataset:
    def __init__(self, df):
        self.fens = torch.Tensor([*map(data_manip.convert_to_bitboard, df['fen'])])
        self.evals = torch.Tensor(df['Evaluation']).reshape(-1,1)

        if torch.cuda.is_available():
            self.fens = self.fens.cuda()
            self.evals = self.evals.cuda()

    def __len__(self):
        return len(self.evals)

    def __getitem__(self, idx):
        return {'input': self.fens[idx], 'output': self.evals[idx]}

