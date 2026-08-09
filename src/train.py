import json
import numpy as np
import torch
import torch.nn as nn
import pickle
import sys
import os

# Fix path so it finds nltk_utils
sys.path.append(os.path.abspath('src'))

from nltk_utils import nlp, tfidf, fit_tfidf, get_hybrid_vector
from model import NeuralNet

# ==========================================
# STEP 1: LOAD AND PREPARE THE DATA
# ==========================================

with open('data/DataSet.json', 'r') as f:
    intents = json.load(f)

tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        xy.append((pattern, tag))

tags = sorted(set(tags))

# --- FIT THE TF-IDF VECTORIZER ---
all_sentences = [pattern for pattern, tag in xy]
fit_tfidf(all_sentences)

# Create the Training Data (X and y)
X_train = []
y_train = []

for (sentence, tag) in xy:
    vector = get_hybrid_vector(sentence, tfidf)
    X_train.append(vector)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Total patterns (sentences): {len(X_train)}")
print(f"Vector size (input neurons): {len(X_train[0])}")
print(f"Total tags (output neurons): {len(tags)}")

# ==========================================
# STEP 2: DEFINE HYPERPARAMETERS
# ==========================================
INPUT_SIZE = len(X_train[0])  
HIDDEN_SIZE = 32              
OUTPUT_SIZE = len(tags)       
LEARNING_RATE = 0.001
EPOCHS = 1500                 

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
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# ==========================================
# STEP 5: SAVE THE MODEL AND VECTORIZER
# ==========================================

# 1. Save the TF-IDF object to a pickle file
with open('saved_models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# 2. Save the PyTorch model data
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
print("TF-IDF Vectorizer saved to saved_models/tfidf_vectorizer.pkl")