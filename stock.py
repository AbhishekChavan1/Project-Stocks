import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

company = input("Enter the company ticket: ")
ticker = yf.Ticker(company)
hist = ticker.history(start="2018-01-01", end=datetime.now())

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(hist["Close"].values.reshape(-1, 1))

prediction_days = int(input("Enter the number of days to predict: "))

x_train = []
y_train = []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], prediction_days))

class Transformer(nn.Module):
    def __init__(self, input_size, prediction_days, dim_feedforward):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward*2)
        self.fc3 = nn.Linear(dim_feedforward*2, dim_feedforward*4)
        self.fc4 = nn.Linear(dim_feedforward*4, dim_feedforward*8)
        self.fc5 = nn.Linear(dim_feedforward*8, dim_feedforward*16)
        self.fc6 = nn.Linear(dim_feedforward*16, dim_feedforward*32)
        self.fc7 = nn.Linear(dim_feedforward*32, dim_feedforward*64)
        self.fc8 = nn.Linear(dim_feedforward*64, dim_feedforward*128)
        self.fc9 = nn.Linear(dim_feedforward*128, dim_feedforward*256)
        self.fc10 = nn.Linear(dim_feedforward*256, prediction_days)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc9(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc10(x)

        return x


model = Transformer(input_size=x_train.shape[1], prediction_days=1, dim_feedforward=22)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
 inputs = torch.from_numpy(x_train).float()
 labels = torch.from_numpy(y_train).float().unsqueeze(1)

optimizer.zero_grad()

outputs = model(inputs)
loss = criterion(outputs, labels)

loss.backward()
optimizer.step()

if (epoch+1) % 10 == 0:
    print("Epoch: {}/{} | Loss: {:.4f}".format(epoch+1, epochs, loss.item()))

future_prediction = []
last_x = scaled_data[-prediction_days:]

for i in range(prediction_days):
  future_input = torch.from_numpy(last_x).float().reshape(1, prediction_days)
  future_price = model(future_input)
  future_prediction.append(future_price.detach().numpy()[0][0])
  last_x = np.append(last_x[1:], future_price.detach().numpy().reshape(-1, 1))

prediction = scaler.inverse_transform(np.array(future_prediction).reshape(-1, 1))

print("Future prices: ")
for i, price in enumerate(prediction):
 print("Día {}: {:.2f}".format(i+1, price[0]))