from flask import Flask, request, jsonify
import torch
import random
from src.nltk_utils import get_sentence_vector
from src.model import NeuralNet
from responses import responses

app = Flask(__name__)

# ==========================================
# 1. LOAD THE AI MODEL
# ==========================================
FILE = "saved_models/trained_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# ==========================================
# 2. CREATE THE API ENDPOINT
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data sent from React/Fastify
    req_data = request.get_json(force=True)
    sentence = req_data.get('sentence', '')

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Process the sentence and predict
    X = get_sentence_vector(sentence)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    # Pick a random response for the predicted tag
    if prob > 0.7:
        bot_response = random.choice(responses[tag])
    else:
        bot_response = "I'm sorry, I didn't quite understand that. Could you rephrase?"

    # Return the tag, confidence, AND the actual response text
    return jsonify({
        "tag": tag,
        "confidence": f"{prob * 100:.2f}%",
        "response": bot_response
    })

# ==========================================
# 3. RUN THE FLASK SERVER
# ==========================================
if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)