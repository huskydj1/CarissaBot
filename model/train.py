import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import chess
import torch
from tqdm import tqdm

import data_manip
from chess_dataset import ChessDataset
from model import CarissaNet

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7" # TODO: add all GPUs to make parallelism work 

fast = False

dataset = ChessDataset({"file_path": 'data/newChessData.csv', "transform": True, "save": True, "fast": fast})

train_len = int(len(dataset)*0.8)
test_len = len(dataset) - train_len

data_train, data_test = torch.utils.data.random_split(dataset, [train_len, test_len])

train_loader = torch.utils.data.DataLoader(data_train, batch_size=2048, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=2048, shuffle=True)

if not fast:
    model = torch.nn.DataParallel(CarissaNet(input_channels=29, blocks=10, filters=128))
else:
    model = torch.nn.DataParallel(CarissaNet(input_channels=25, blocks=10, filters=128))

if torch.cuda.is_available():
    model = model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

train_losses = []
test_losses = []

for epoch in range(1, 61):
    model.train()
    sum_loss = 0
    for elem in tqdm(train_loader):
        optimizer.zero_grad()

        data_input = elem['input']
        data_output = elem['output']
        if torch.cuda.is_available():
            data_input = data_input.cuda()
            data_output = data_output.cuda()

        output = model(data_input)
        loss = loss_fn(output, data_output)
        sum_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optimizer.step()

    avg_loss = sum_loss / len(train_loader)
    print(f'Average Loss Epoch {epoch:5}: {avg_loss:10}')
    train_losses.append(avg_loss)

    with torch.no_grad():
        model.eval()
        sum_loss = 0
        for elem in tqdm(test_loader):
            data_input = elem['input']
            data_output = elem['output']
            if torch.cuda.is_available():
                data_input = data_input.cuda()
                data_output = data_output.cuda()
            output = model(data_input)
            loss = loss_fn(output, data_output)
            sum_loss += loss.item()

        avg_loss = sum_loss / len(test_loader)
        print(f'Average TEST Loss Epoch {epoch:5}: {avg_loss:10}')
        test_losses.append(avg_loss)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'model_111821_newdata_10x128_{epoch}.pt')

x = list(range(1, 61))

plt.plot(x, train_losses, label='Train Loss')
plt.plot(x, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig('loss_plot_111821_newdata_10x128.png')
