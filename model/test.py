import pandas as pd
import numpy as np
import torch

import data_manip

df = pd.read_csv('data/tinyChessData.csv')

df['FEN'] = df['FEN'].apply(data_manip.convert_to_bitboard)
print(df['FEN'])
print(np.array(list(df['FEN'])))
fens = torch.Tensor(np.array(list(df['FEN'])))

print(fens.size())

