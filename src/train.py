import json
import numpy as np
import torch
import torch.nn as nn
from nltk_utils import get_sentence_vector
from model import NeuralNet

# ==========================================
# STEP 1: LOAD AND PREPARE THE DATA
# ==========================================
print("Loading BERT model and encoding sentences... (This might take a minute)")

with open('data/DataSet.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Get the 384-dimensional BERT vector
        vector = get_sentence_vector(pattern)
        xy.append((vector, tag))

tags = sorted(set(tags))

X_train = []
y_train = []

for (vector, tag) in xy:
    X_train.append(vector)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Total patterns: {len(X_train)}")
print(f"Vector size (input neurons): {len(X_train[0])}") # Will print 384
print(f"Total tags: {len(tags)}")

# ==========================================
# STEP 2: DEFINE HYPERPARAMETERS
# ==========================================
INPUT_SIZE = 384               # BERT Mini outputs 384 dimensions
HIDDEN_SIZE = 64               
OUTPUT_SIZE = len(tags)        
LEARNING_RATE = 0.001
EPOCHS = 500                   # BERT needs fewer epochs to learn perfectly

# ==========================================
# STEP 3: SETUP PyTorch
# ==========================================
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# STEP 4: THE TRAINING LOOP
# ==========================================
print("\nStarting Training...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# ==========================================
# STEP 5: SAVE THE MODEL
# ==========================================
data = {
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "tags": tags
}

FILE = "saved_models/trained_model.pth"
torch.save(data, FILE)

print(f"\nTraining complete. Model saved to {FILE}")