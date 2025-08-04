"""
Train, Test and Predict Model
Integrate Vector Normalization
"""

# Min-Max Normalization - GLOBALS Reuse in Normalization Function
min_vals = None
max_vals = None

def initialize_min_max(features):
    global min_vals, max_vals
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)

def min_max_normalize(X):
    global min_vals, max_vals
    if min_vals is not None and max_vals is not None:
        return (X - min_vals) / (max_vals - min_vals + 1e-8) # eps=1e-8 - avoid divison by zero
    raise ValueError("min_vals and max_vals are not initialized")

