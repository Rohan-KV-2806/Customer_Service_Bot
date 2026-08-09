import torch
from src.nltk_utils import tokenize, bag_of_words
from src.model import NeuralNet

# 1. Load the saved model
FILE = "saved_models/trained_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# 2. Define 50 Test Cases
# Format: ("Sentence a user might type", "The Tag it SHOULD predict")
test_data = [
    # Greeting (5)
    ("Hi there", "Greeting"),
    ("Hello bot", "Greeting"),
    ("Good morning", "Greeting"),
    ("Hey", "Greeting"),
    ("Sup", "Greeting"),
    
    # Goodbye (5)
    ("Bye bye", "Goodbye"),
    ("See you later", "Goodbye"),
    ("I'm leaving now", "Goodbye"),
    ("Catch you later", "Goodbye"),
    ("Goodnight", "Goodbye"),
    
    # Request_Refund (5)
    ("I need my money back", "Request_Refund"),
    ("Give me a refund", "Request_Refund"),
    ("Can I get refunded for this?", "Request_Refund"),
    ("I want to return this for cash", "Request_Refund"),
    ("Refund my account", "Request_Refund"),
    
    # Request_Replacement (5)
    ("Send me a new one", "Request_Replacement"),
    ("This is broken, I want another", "Request_Replacement"),
    ("Can I swap this?", "Request_Replacement"),
    ("I need to replace my order", "Request_Replacement"),
    ("Give me a replacement", "Request_Replacement"),
    
    # Cancel_Order (5)
    ("Stop my order", "Cancel_Order"),
    ("I need to cancel this", "Cancel_Order"),
    ("Don't ship that anymore", "Cancel_Order"),
    ("Cancel order 4", "Cancel_Order"),
    ("How do I cancel?", "Cancel_Order"),
    
    # Track_Shipment (5)
    ("Where is my package?", "Track_Shipment"),
    ("When will this arrive?", "Track_Shipment"),
    ("Tracking number please", "Track_Shipment"),
    ("Is my order shipped yet?", "Track_Shipment"),
    ("How long until delivery?", "Track_Shipment"),
    
    # Speak_To_Human (5)
    ("Let me talk to a person", "Speak_To_Human"),
    ("Connect me to an agent", "Speak_To_Human"),
    ("I want a real human", "Speak_To_Human"),
    ("Get me a manager", "Speak_To_Human"),
    ("Talk to support staff", "Speak_To_Human"),
    
    # Password_Reset (5)
    ("I forgot my password", "Password_Reset"),
    ("How do I reset my login?", "Password_Reset"),
    ("Need to change my password", "Password_Reset"),
    ("Password reset link", "Password_Reset"),
    ("I can't remember my password", "Password_Reset"),
    
    # Thanks (5)
    ("Thank you", "Thanks"),
    ("Thanks a lot", "Thanks"),
    ("I appreciate it", "Thanks"),
    ("Much obliged", "Thanks"),
    ("Cool thanks", "Thanks"),
    
    # Bot_Identity (5)
    ("Who are you?", "Bot_Identity"),
    ("What is your name?", "Bot_Identity"),
    ("Are you a robot?", "Bot_Identity"),
    ("What can you do?", "Bot_Identity"),
    ("Are you AI?", "Bot_Identity")
]

# 3. Run the tests
correct = 0
total = len(test_data)

print("\n===== RUNNING TEST CASES =====\n")

for sentence, expected_tag in test_data:
    # Process the sentence
    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
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