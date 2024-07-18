import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leakyrelu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leakyrelu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leakyrelu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leakyrelu2(x)
        x = self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)
    
    