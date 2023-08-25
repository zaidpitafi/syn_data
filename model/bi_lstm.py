import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 256, num_layers = 6):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        return out

# # Example usage
# input_size = 10
# hidden_size = 128
# num_layers = 2
# num_classes = 3

# # Create an instance of the BiLSTM model
# model = BiLSTMModel(input_size, hidden_size, num_layers, num_classes)

# # Create a random input tensor
# batch_size = 16
# sequence_length = 20
# input_tensor = torch.randn(batch_size, sequence_length, input_size)

# # Forward pass
# output = model(input_tensor)
# print("Output shape:", output.shape)
