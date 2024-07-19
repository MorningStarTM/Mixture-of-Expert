import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.softmax(self.layer2(x), dim=1)
    
    def train_expert(self, x_train, y_train, epochs=500, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
    

    def save_model(self, path:str):
        torch.save(self.state_dict(), path)

    def load_model(self, path:str):
        self.load_state_dict(torch.load(path))



