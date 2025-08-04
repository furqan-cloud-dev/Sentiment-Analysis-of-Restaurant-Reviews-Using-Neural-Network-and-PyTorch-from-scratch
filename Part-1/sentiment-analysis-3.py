
"""
Simple MLP Classifier - Multi-Layer Perceptron

Inheritance/Sub-Class from PyTorch's: torch.nn.Module (Base Class Of Neural Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample input vector
features = [0, 0, 0, 9, 1, 1, 3]
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(7, 16)   # First hidden layer
        self.fc2 = nn.Linear(16, 8)   # Second hidden layer
        self.fc3 = nn.Linear(8, 4)    # Output layer for 4 sentiment classes

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation after first hidden layer
        x = F.relu(self.fc2(x))  # Activation after second hidden layer
        x = self.fc3(x)          # Output logits (no softmax for CrossEntropyLoss)
        return x

# Instantiate and run
model = MLPClassifier()
output = model(features_tensor)

# Predicted class
predicted_class = torch.argmax(output, dim=1)

print("Logits:", output)
print("Predicted class:", predicted_class.item())


# Train This Model - Supervised Learning

# Sample target (e.g., class label index)
target = torch.tensor([2])  # Assuming class '2' is the correct label (positive)

# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training step
optimizer.zero_grad()
loss = loss_fn(output, target)
loss.backward()
optimizer.step()

print("Training loss:", loss.item())
