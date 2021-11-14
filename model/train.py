import pandas as pd
import chess

import data_manip
from chess_dataset import ChessDataset

df = pd.read_csv('data/smallerChessData2.csv')
df['Evaluation'] = df['Evaluation'].apply(data_manip.eval_to_pawn).apply(data_manip.pawn_to_prob)

dataset = ChessDataset(df)

