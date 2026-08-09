import torch
import pickle
import sys
import os

# Fix path so it finds nltk_utils inside the src folder
sys.path.append(os.path.abspath('src'))

from nltk_utils import get_hybrid_vector
from model import NeuralNet

# 1. Load the saved PyTorch model
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

# 2. Load the saved TF-IDF Vectorizer
with open('saved_models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 3. Define 50 Test Cases
test_data = [
    ("Hi there", "Greeting"), ("Hello bot", "Greeting"), ("Good morning", "Greeting"),
    ("Hey", "Greeting"), ("Sup", "Greeting"), ("Bye bye", "Goodbye"), ("See you later", "Goodbye"),
    ("I'm leaving now", "Goodbye"), ("Catch you later", "Goodbye"), ("Goodnight", "Goodbye"),
    ("I need my money back", "Request_Refund"), ("Give me a refund", "Request_Refund"),
    ("Can I get refunded for this?", "Request_Refund"), ("I want to return this for cash", "Request_Refund"),
    ("Refund my account", "Request_Refund"), ("Send me a new one", "Request_Replacement"),
    ("This is broken, I want another", "Request_Replacement"), ("Can I swap this?", "Request_Replacement"),
    ("I need to replace my order", "Request_Replacement"), ("Give me a replacement", "Request_Replacement"),
    ("Stop my order", "Cancel_Order"), ("I need to cancel this", "Cancel_Order"),
    ("Don't ship that anymore", "Cancel_Order"), ("Cancel order 4", "Cancel_Order"),
    ("How do I cancel?", "Cancel_Order"), ("Where is my package?", "Track_Shipment"),
    ("When will this arrive?", "Track_Shipment"), ("Tracking number please", "Track_Shipment"),
    ("Is my order shipped yet?", "Track_Shipment"), ("How long until delivery?", "Track_Shipment"),
    ("Let me talk to a person", "Speak_To_Human"), ("Connect me to an agent", "Speak_To_Human"),
    ("I want a real human", "Speak_To_Human"), ("Get me a manager", "Speak_To_Human"),
    ("Talk to support staff", "Speak_To_Human"), ("I forgot my password", "Password_Reset"),
    ("How do I reset my login?", "Password_Reset"), ("Need to change my password", "Password_Reset"),
    ("Password reset link", "Password_Reset"), ("I can't remember my password", "Password_Reset"),
    ("Thank you", "Thanks"), ("Thanks a lot", "Thanks"), ("I appreciate it", "Thanks"),
    ("Much obliged", "Thanks"), ("Cool thanks", "Thanks"), ("Who are you?", "Bot_Identity"),
    ("What is your name?", "Bot_Identity"), ("Are you a robot?", "Bot_Identity"),
    ("What can you do?", "Bot_Identity"), ("Are you AI?", "Bot_Identity")
]

# 4. Run the tests
correct = 0
total = len(test_data)

print("\n===== RUNNING TEST CASES =====\n")

for sentence, expected_tag in test_data:
    # Get the hybrid vector using the LOADED tfidf_vectorizer
    X = get_hybrid_vector(sentence, tfidf_vectorizer)
    X = torch.from_numpy(X).float().unsqueeze(0)
    
    # Predict
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    actual_tag = tags[predicted.item()]
    
    # Check confidence
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()
    
    # Grade it
    if actual_tag == expected_tag:
        status = "✅ PASS"
        correct += 1
    else:
        status = "❌ FAIL"
        
    print(f"{status} | Expected: {expected_tag:20} | Got: {actual_tag:20} | Conf: {prob*100:.1f}% | Sentence: '{sentence}'")

print(f"\n===== FINAL SCORE: {correct}/{total} =====")