# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Stock price prediction is an important task in financial analysis. The goal is to predict future stock prices based on historical data. In this experiment, historical stock price data is used to train a Recurrent Neural Network model. The dataset contains stock information such as the closing price over time. The model learns patterns from past prices and predicts future stock prices.

## Design Steps

### Step 1:

Load the stock price dataset and select the closing price values. Normalize the data using MinMaxScaler to scale the values between 0 and 1 for better training performance.

### Step 2:

Convert the time-series data into sequences so that the RNN can learn temporal patterns. Each sequence contains a fixed number of previous time steps used to predict the next stock price.

### Step 3:

Define the RNN model using PyTorch, train the model using the training dataset, and evaluate its performance by predicting stock prices on the test dataset. Finally, compare the actual stock prices and predicted prices using a plot.

#### Name: Mithun Kumar G
#### Register Number: 212224230160
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="703" height="511" alt="image" src="https://github.com/user-attachments/assets/2e197bd0-7fc8-4b84-9d69-9ab3c8af66f6" />

### Predictions 

<img width="768" height="526" alt="image" src="https://github.com/user-attachments/assets/96c8ccfb-7212-4357-99a9-c92929dc16e0" />

<img width="252" height="52" alt="image" src="https://github.com/user-attachments/assets/0c694315-ee3a-4c40-8773-380b8e647809" />

## Result

Thus, a Recurrent Neural Network (RNN) model was successfully developed for stock price prediction. The model was trained using historical stock price data and the predicted prices were compared with the actual prices using a graph, demonstrating the model’s ability to learn patterns in time-series data.
