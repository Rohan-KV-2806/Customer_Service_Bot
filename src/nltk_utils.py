import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading spaCy language model...")
nlp = spacy.load("en_core_web_md")

# No custom tokenizer anymore! This prevents the pickle error.
tfidf = TfidfVectorizer(lowercase=True)

def fit_tfidf(all_sentences):
    """Learns the vocabulary and IDF values from all training sentences"""
    tfidf.fit(all_sentences)

def get_hybrid_vector(sentence, tfidf_vectorizer):
    """
    Combines TF-IDF (keyword checklist) and spaCy (semantic meaning)
    into a single 1D numpy array.
    """
    # 1. Get the spaCy 300-dimensional vector
    doc = nlp(sentence)
    spacy_vec = doc.vector
    
    # 2. Get the TF-IDF vector (using the passed-in fitted vectorizer)
    tfidf_vec = tfidf_vectorizer.transform([sentence]).toarray()[0]
    
    # 3. Staple them together (np.concatenate)
    hybrid_vec = np.concatenate((tfidf_vec, spacy_vec))
    
    return hybrid_vec.astype(np.float32)