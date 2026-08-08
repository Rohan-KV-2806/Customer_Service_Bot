import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')

stemmer = PorterStemmer()

def stem(word):
    return stemmer.stem(word.lower())

def tokenize(sentence):
    tokens = word_tokenize(sentence)
    return [stem(w) for w in tokens]

def bag_of_words(sentence, all_words):
    tokens = tokenize(sentence)
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for idx, w in enumerate(all_words):
        if w in tokens:
            bag[idx] = 1.0
            
    return bag

