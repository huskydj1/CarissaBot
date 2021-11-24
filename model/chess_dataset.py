import chess
import numpy as np
import pandas as pd
import os
import json
import torch
from tqdm import tqdm

import data_manip

tqdm.pandas()

class ChessDataset:
    def __init__(self, args):
        df = pd.read_csv(args["file_path"])
        if args["transform"]:
            df['Evaluation'] = df['Evaluation'].progress_apply(data_manip.eval_to_pawn).apply(data_manip.pawn_to_prob)

        cache_dir = "data/cached_data." if not args["fast"] else "data/cached_data_fast."

        if os.path.isfile(cache_dir + "args") and json.load(open(cache_dir + "args", "r")) == args:
           load = torch.load(cache_dir + "pt")
           self.fens = load["fens"]
           self.evals = load["evals"]
        else:
            if not args["fast"]:
                df['FEN'] = df['FEN'].progress_apply(data_manip.convert_to_bitboard)
                self.fens = torch.Tensor(np.array(list(df['FEN'])))
                print(self.fens.size(), "fens size")
            else:
                self.fens = data_manip.convert_to_bitboard_fast(df['FEN'].tolist())
            self.evals = torch.Tensor(df['Evaluation']).reshape(-1,1)

            if args["save"]:
                torch.save({
                    "fens": self.fens,
                    "evals": self.evals
                }, cache_dir + "pt")
                with open(cache_dir + "args", "w") as f:
                    json.dump(args, f)

    def __len__(self):
        return len(self.evals)

    def __getitem__(self, idx):
        return {'input': self.fens[idx], 'output': self.evals[idx]}

