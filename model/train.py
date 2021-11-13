import pandas as pd
import chess

import data_manip
from chess_dataset import ChessDataset
from dbn import DBN

df = pd.read_csv('data/smallerChessData2.csv')
df['Evaluation'] = df['Evaluation'].apply(data_manip.eval_to_pawn).apply(data_manip.pawn_to_prob)

dataset = ChessDataset(df)

layers = [1000,800,600,400,200,100]
model = DBN(1856, layers)

model.train_DBN(dataset, epochs=100, batch_size=128, learning_rate=1e-3)

