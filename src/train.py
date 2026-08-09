import json
import numpy as np
import torch
import torch.nn as nn
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet

# ==========================================
# STEP 1: LOAD AND PREPARE THE DATA
# ==========================================

with open('data/DataSet.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each intent in the JSON
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        # Tokenize the pattern (this also stems and lowercases it)
        tokenized_pattern = tokenize(pattern)
        # Add all words to our master vocabulary
        all_words.extend(tokenized_pattern)
        # Save the pair so we remember which sentence belongs to which tag
        xy.append((tokenized_pattern, tag))

# Remove duplicates and sort alphabetically
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create the Training Data (X and y)
X_train = []
y_train = []

for (tokenized_pattern, tag) in xy:
    # Create the 0s and 1s array for this sentence
    bow = bag_of_words(tokenized_pattern, all_words)
    X_train.append(bow)
    # Find the index of the tag (e.g., 0 for Greeting, 1 for Goodbye)
    label = tags.index(tag)
    y_train.append(label)

# Convert from Python lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Total patterns (sentences): {len(X_train)}")
print(f"Vocabulary size (input neurons): {len(all_words)}")
print(f"Total tags (output neurons): {len(tags)}")

# ==========================================
# STEP 2: DEFINE HYPERPARAMETERS
# ==========================================
INPUT_SIZE = len(X_train[0])  # The length of our Bag of Words array
HIDDEN_SIZE = 16              # Number of neurons in hidden layers
OUTPUT_SIZE = len(tags)       # 30 tags
LEARNING_RATE = 0.001
EPOCHS = 1000                 # How many times we loop through the data
BATCH_SIZE = 16               # How many sentences to process at once

# ==========================================
# STEP 3: SETUP PyTorch
# ==========================================

# Convert NumPy arrays to PyTorch Tensors
# X must be float32, y must be long (int64) for PyTorch's loss function
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

# Initialize the Model
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Loss Function (CrossEntropy is standard for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam adjusts the weights during backpropagation)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# STEP 4: THE TRAINING LOOP
# ==========================================
print("\nStarting Training...")

for epoch in range(EPOCHS):
    # 1. Clear old gradients
    optimizer.zero_grad()
    
    # 2. Feed forward (make predictions)
    outputs = model(X_train_tensor)
    
    # 3. Calculate loss (how wrong were the predictions?)
    loss = criterion(outputs, y_train_tensor)
    
    # 4. Backpropagation (calculate new gradients)
    loss.backward()
    
    # 5. Update the weights
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# ==========================================
# STEP 5: SAVE THE MODEL
# ==========================================

# We save the model weights AND the vocabulary/tags because we need them
# to process new user inputs in our Flask API later.
data = {
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words": all_words,
    "tags": tags
}

FILE = "saved_models/trained_model.pth"
torch.save(data, FILE)

print(f"\nTraining complete. Model saved to {FILE}")