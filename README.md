# Customer Service Bot

An AI-powered intent classification system that analyzes customer support queries and categorizes them into distinct intents. By leveraging transformer-based sentence embeddings, the model understands the semantic meaning of user messages rather than just matching keywords.

### Tools Used
* **Python** (Core logic)
* **PyTorch** (Deep Learning framework)
* **HuggingFace Transformers** (BERT model for sentence embeddings)
* **Flask** (Backend API)
* **JSON** (Dataset storage)

### What It Does
1. Takes a user's raw text (e.g., *"I need my money back"*).
2. Converts the text into a 384-dimensional mathematical vector using a pre-trained BERT model.
3. Feeds the vector into a custom Neural Network.
4. Predicts the user's "Intent" (e.g., Return_Refund) and calculates a confidence score.
5. Selects a context-appropriate response randomly from a predefined list and returns it via a REST API.

### Steps to Use
1. **Clone the repository** and navigate into the folder.
2. **Set up the environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install torch sentence-transformers numpy flask
   ```
3. **Add the data:** Make sure that `DataSet.json` is inside the `data/` folder.
4. **Train the model:**
   ```bash
   python src/train.py
   ```
5. **Start the backend API:**
   ```bash
   python app.py
   ```
6. **Test the API:** Send a POST request to http://localhost:5000/predict with JSON:
   ```bash
   {"sentence": "I need my money back"}
   ```
