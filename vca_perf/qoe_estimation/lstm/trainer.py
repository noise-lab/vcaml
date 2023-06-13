import torch
from torch import nn
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(target).float()
        self.X = torch.tensor(dataframe).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        return self.X[i], self.y[i]

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

class LSTMTrainer:

    def __init__(self):
        pass

    def train_model(self, data_loader, model, loss_function, optimizer):
        num_batches = len(data_loader)
        total_loss = 0
        model.train()
        
        for X, y in data_loader:
            output = model(X)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}")

    def test_model(self, data_loader, model, loss_function):
        
        num_batches = len(data_loader)
        total_loss = 0

        model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                output = model(X)
                total_loss += loss_function(output, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")

    def predict(self, data_loader, model):
        output = torch.tensor([])
        model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                y_star = model(X)
                output = torch.cat((output, y_star), 0)
        
        return output


