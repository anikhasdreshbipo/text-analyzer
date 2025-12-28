#!/usr/bin/env python3
"""
Prediction script for sentiment analysis using the legacy model.
"""

import sys
import numpy as np

# Add the models directory to the path
sys.path.insert(0, 'src/models')

from sentiment_legacy import SentimentLegacyModel

def load_model():
    """Load the legacy sentiment analysis model."""
    model = SentimentLegacyModel()
    # In a real scenario, we would load pre-trained weights
    # model.load('path/to/weights')
    return model

def preprocess_text(text, vocab, max_length=200):
    """Convert text to token indices using a simple vocab mapping.
    This is a dummy implementation; real preprocessing would involve tokenization.
    """
    # Dummy tokenization: split by space and map to indices using vocab
    tokens = text.lower().split()
    indices = [vocab.get(token, 0) for token in tokens[:max_length]]
    # Pad or truncate
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return np.array([indices])

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <text>")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    print(f"Analyzing sentiment for: {text}")
    
    # Dummy vocabulary mapping (in real scenario, load from file)
    vocab = {'good': 1, 'bad': 2, 'great': 3, 'terrible': 4}  # etc.
    
    model = load_model()
    input_array = preprocess_text(text, vocab)
    prediction = model.predict(input_array)
    
    sentiment_score = prediction[0][0]
    sentiment = "positive" if sentiment_score > 0.5 else "negative"
    print(f"Sentiment score: {sentiment_score:.4f}")
    print(f"Predicted sentiment: {sentiment}")

if __name__ == '__main__':
    main()