import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from utils import create_inout_sequences
from lstm import LSTM
import matplotlib.pyplot as plt

from tqdm import tqdm

torch.manual_seed(1)
np.random.seed(1)

df = pd.read_csv("./data/cpu-full-a.csv")
cpu_vals = np.array(df["cpu"].values)


scaler = MinMaxScaler(feature_range=(-1,1))
train_data_normalized = scaler.fit_transform(cpu_vals.reshape(-1,1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
print(len(train_data_normalized))
train_window = 100

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

model = LSTM()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 100

for i in range(epochs):
    for seq, labels in tqdm(train_inout_seq):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    print(f"Epoch {i+1} loss: {single_loss.item()}")

torch.save(model, "./models/lstm_time_series.pth")

fut_pred = 400

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
# print(actual_predictions)

x = np.arange(132, 532, 1)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['cpu'])
plt.plot(x,actual_predictions)
plt.show()