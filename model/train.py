import pandas as pd
import chess
import torch
from tqdm import tqdm

import data_manip
from chess_dataset import ChessDataset
from model import CarissaNet

df = pd.read_csv('data/smallerChessData2.csv')
df['Evaluation'] = df['Evaluation'].apply(data_manip.eval_to_pawn).apply(data_manip.pawn_to_prob)

dataset = ChessDataset(df)

train_len = int(len(dataset)*0.8)
test_len = len(dataset) - train_len

data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])

train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=512, shuffle=True)

model = CarissaNet(blocks=10, filters=128)

if torch.cuda.is_available():
    model = model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(1, 101):
    model.train()
    sum_loss = 0
    for elem in tqdm(train_loader):
        optimizer.zero_grad()

        output = model(elem['input'])
        loss = loss_fn(output, elem['output'])
        sum_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optimizer.step()

    avg_loss = sum_loss / len(train_loader)
    print(f'Average Loss Epoch {epoch:5}: {avg_loss:10}')

    with torch.no_grad():
        model.eval()
        sum_loss = 0
        for elem in tqdm(test_loader):
            output = model(elem['input'])
            loss = loss_fn(output, elem['output'])
            sum_loss += loss.item()

        avg_loss = sum_loss / len(test_loader)
        print(f'Average TEST Loss Epoch {epoch:5}: {avg_loss:10}')

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'model_{epoch}.pt')