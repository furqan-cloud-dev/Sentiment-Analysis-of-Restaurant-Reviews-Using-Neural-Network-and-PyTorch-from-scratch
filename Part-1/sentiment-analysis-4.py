"""
Building Neural Network
"""

# Neural Network - Sentiment Classifier
import torch.nn as nn
import torch.nn.functional as F

# Inheritance/Sub-Class from PyTorch's: torch.nn.Module (Base Class Of Neural Network)
class SentimentClassifierNN(nn.Module):
    def __init__(self):
        super(SentimentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(7, 16)     # Input: 7 features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)     # Output: 4 classes

    # Must implement a forward(input) method
    # Transforming the input tensor(s) into output tensor(s)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)  # Softmax Activation for Probability distribution



# Create an instance of the Neural Network
model = SentimentClassifierNN()

# Create a dummy input
input_tensor = torch.randn(1, 7) # Batch size 1, input features 7

# Pass input through the network
output = model(input_tensor)
print(output)