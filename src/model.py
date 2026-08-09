import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # Layer 1: Input layer (Bag of Words) -> Hidden Layer
        self.l1 = nn.Linear(input_size, hidden_size) 
        
        # Layer 2: Hidden Layer -> Hidden Layer
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        
        # Layer 3: Hidden Layer -> Output Layer (30 tags)
        self.l3 = nn.Linear(hidden_size, num_classes) 
        
        # Activation function to bend the math (allows non-linear patterns)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass data through layer 1, then activate
        out = self.l1(x)
        out = self.relu(out)
        
        # Pass data through layer 2, then activate
        out = self.l2(out)
        out = self.relu(out)
        
        # Pass data through layer 3 (No activation here, PyTorch's loss function handles it)
        out = self.l3(out)
        
        return out