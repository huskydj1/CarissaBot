import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_smaller = df.sample(frac = 0.16)

df_random = pd.read_csv('data/random_evals.csv')

df_new = pd.concat([df_smaller, df_random])

df_new.to_csv('data/newChessData.csv')
