import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, sequence_length, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Apply CNN layers
        x = self.cnn(x)
        
        # Reshape output for LSTM input
        x = x.permute(0, 2, 1)  # Permute dimensions to (batch_size, sequence_length, input_size)
        
        # Apply LSTM layer
        output, _ = self.lstm(x)
        x = output[:, -1, :]  # Extract output of the last time step
        
        # Apply fully connected layer
        x = self.fc(x)
        
        return x
