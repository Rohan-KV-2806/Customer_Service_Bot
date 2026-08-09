import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Only download once
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()

def tokenize(sentence):
    # Split into words, lowercase them, and chop off punctuation
    tokens = word_tokenize(sentence)
    return [stemmer.stem(w.lower()) for w in tokens if w.isalpha()]

def bag_of_words(tokenized_sentence, all_words):
    # Create an array of zeros matching the size of our vocabulary
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # For each word in the sentence, if it's in our vocab, change 0 to 1
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag