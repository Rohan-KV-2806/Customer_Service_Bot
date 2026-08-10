from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# Load the pre-trained BERT model.
# This will download a ~80MB model the first time you run it.
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_vector(sentence):
    """
    Converts a sentence into a 384-dimensional dense vector using BERT.
    This model understands word ORDER and CONTEXT, unlike spaCy.
    """
    vector = encoder.encode(sentence)
    return vector.astype(np.float32)