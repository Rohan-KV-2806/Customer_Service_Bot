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

# 2. Rebuild the model and load the trained brain
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

print("\n==================================")
print("Tag Predictor is online!")
print("Type 'quit' to exit.")
print("==================================\n")

# 3. The Chat Loop
while True:
    sentence = input("You: ")
    if sentence.lower() == 'quit':
        break

    # Get the 384-dimensional BERT vector
    X = get_sentence_vector(sentence)
    # Convert to tensor and add a batch dimension
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Feed it to the model
    output = model(X)
    
    # Get the predicted tag index
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check the model's confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    # Just print the tag and confidence
    print(f" -> Predicted Tag: {tag} ({prob*100:.1f}% confidence)\n")