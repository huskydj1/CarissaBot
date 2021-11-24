import pandas as pd

df = pd.read_csv('data/chessData.csv')

df_new = df.sample(frac = 0.000008)

# df_random = pd.read_csv('data/random_evals.csv')

# df_new = pd.concat([df, df_random])

df_new.to_csv('data/tinyChessData.csv')
