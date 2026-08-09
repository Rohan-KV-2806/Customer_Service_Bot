import json
import random
import torch
from src.nltk_utils import tokenize, bag_of_words
from src.model import NeuralNet

# 1. Load the saved model data
FILE = "saved_models/trained_model.pth"
data = torch.load(FILE)

# Extract the settings and the trained weights
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# 2. Rebuild the model and load the trained brain into it
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
# Put the model in evaluation mode (turns off dropout/batch norm stuff we didn't use, but good practice)
model.eval()

# 3. Load the JSON again so we can grab the responses
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

    # Step A: Clean the user's input (tokenize & stem)
    tokenized_sentence = tokenize(sentence)
    
    # Step B: Turn input into 0s and 1s (Bag of Words)
    # PyTorch expects a tensor, and we add a fake batch dimension [1, vocab_size]
    X = bag_of_words(tokenized_sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Step C: Feed it to the model
    output = model(X)
    
    # Step D: Get the prediction
    # The output is 30 raw numbers. We use softmax to turn them into percentages.
    # Then argmax to find the index of the highest percentage.
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Step E: Check how confident the model is
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Step F: Respond
    if prob.item() > 0.7: # 70% confidence threshold
        for intent in intents['intents']:
            if intent['tag'] == tag:
                # Pick a random response from the JSON for this tag
                print(f"Bot: {random.choice(intent['responses'])}")
    else:
        print("Bot: I'm sorry, I didn't quite understand that. Could you rephrase?")