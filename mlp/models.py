import torch
from torch import nn

class Perceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    
    def forward(self, x):
        fc_output = self.fc(x)
        output = self.relu(fc_output) # instead of Heaviside step fn
        return output
