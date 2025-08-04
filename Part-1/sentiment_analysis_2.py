"""
Tensors & PyTorch
"""

# imports the PyTorch library
import torch

# words count against each category of text review: positive, negative etc.
features = [0, 0, 0, 9, 1, 1, 3]

# creates a tensor from the provided list or array-like structure.
features_tensor = torch.tensor(features)

print(features_tensor)
print("Tensor shape:", features_tensor.shape)
print("Tensor type:", features_tensor.dtype)


# Just comment the next line to move forward
exit()
###########################

import torch.nn as nn
# Create a Linear layer
linear_layer = nn.Linear(
                         in_features=7,
                         out_features=4
                         )

# Simply convert the tensor to float before passing it to the linear layer:
features_tensor = torch.tensor(features, dtype=torch.float32)

# nn.Linear expects a 2D input tensor of shape [batch_size, in_features],
# But our current features_tensor is 1D with shape [7]. So, reshape it
features_tensor = features_tensor.unsqueeze(0)  # Shape becomes [1, 7]

# Pass input_tensor through the linear layer
output = linear_layer(features_tensor)
print("Output: ",output)

# This gives you the index of the largest logit, i.e., the class with the highest predicted score.
predicted_class = torch.argmax(output, dim=1)
print("predicted_class: ", predicted_class)

# For training, we'd typically use:
loss_fn = nn.CrossEntropyLoss()
target_class = torch.tensor([2]) # positive class category index: 2
loss = loss_fn(output, target_class)

print("Loss: ", loss)


# Just comment the next line to move forward
exit()
##################################################33

# multi-layer architecture
model = nn.Sequential(
    nn.Linear(7, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 4)
)

output = model(features_tensor)
print("output: ", output)

predicted_class = torch.argmax(output, dim=1)
print("predicted_class: ", predicted_class)

# For training, we'd typically use:
loss_fn = nn.CrossEntropyLoss()
target_class = torch.tensor([2]) # positive class category index: 2
loss = loss_fn(output, target_class)

print("Loss: ", loss)