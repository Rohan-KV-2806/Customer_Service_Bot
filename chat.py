import json
import random
import torch
from src.nltk_utils import get_sentence_vector
from src.model import NeuralNet

# 1. Load the saved model data
FILE = "saved_models/trained_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data["tags"]
model_state = data["model_state"]

# 2. Rebuild the model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# 3. Load the JSON for responses
with open('data/DataSet.json', 'r') as f:
    intents = json.load(f)

print("\n==================================")
print("Customer Service Bot is online!")
print("Type 'quit' to exit.")
print("==================================\n")

# 4. The Chat Loop
while True:
    sentence = input("You: ")
    if sentence.lower() == 'quit':
        break

    # Get the 300-dimensional vector for the user's input
    X = get_sentence_vector(sentence)
    # Convert to tensor and add fake batch dimension
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Feed it to the model
    output = model(X)
    
    # Get the prediction
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Respond
    if prob.item() > 0.7:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                print(f"Bot: {random.choice(intent['responses'])}")
    else:
        print("Bot: I'm sorry, I didn't quite understand that. Could you rephrase?")