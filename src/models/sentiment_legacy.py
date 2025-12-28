"""
Modernized sentiment analysis model using TensorFlow 2.x Keras API.
This refactored version replaces the legacy TensorFlow 1.x patterns with clean,
maintainable code and comprehensive inline documentation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


class SentimentLegacyModel:
    """
    A modernized sentiment analysis model using word embeddings and dense layers.
    
    This model uses the Keras functional API to define a neural network that
    processes integer-encoded text sequences and outputs a sentiment score
    between 0 (negative) and 1 (positive).
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding vectors.
        max_length (int): Maximum length of input sequences.
        model (tf.keras.Model): The compiled Keras model.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        """
        Initialize the sentiment model with the given hyperparameters.
        
        Args:
            vocab_size: Number of unique tokens in the vocabulary.
            embedding_dim: Dimension of the embedding space.
            max_length: Maximum sequence length (shorter sequences are padded,
                        longer sequences are truncated).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = self._build_model()
        self._compile_model()
    
    def _build_model(self):
        """
        Construct the neural network architecture using Keras functional API.
        
        Architecture:
        1. Input layer for integer‑encoded sequences of length `max_length`.
        2. Embedding layer that maps each token to a dense vector of size
           `embedding_dim`. The embeddings are trainable.
        3. Flatten layer to convert the 2D sequence of embeddings into a 1D
           vector (size = max_length * embedding_dim).
        4. Two fully‑connected (Dense) layers with ReLU activation.
        5. Output layer with a single neuron and sigmoid activation, producing
           a sentiment probability.
        
        Returns:
            tf.keras.Model: An uncompiled Keras model.
        """
        # Input layer: integer sequences of fixed length
        inputs = layers.Input(shape=(self.max_length,), dtype='int32', name='input')
        
        # Embedding layer: learn dense representations for each token
        # The embedding matrix has shape (vocab_size, embedding_dim)
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # Flatten the 2D sequence of embeddings into a 1D feature vector
        x = layers.Flatten(name='flatten')(x)
        
        # First hidden dense layer with 256 units and ReLU activation
        x = layers.Dense(256, activation='relu', name='dense1')(x)
        
        # Second hidden dense layer with 128 units and ReLU activation
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        
        # Output layer: single neuron with sigmoid activation for binary sentiment
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create the Keras Model instance
        model = models.Model(inputs=inputs, outputs=outputs, name='sentiment_model')
        return model
    
    def _compile_model(self):
        """
        Compile the model with an appropriate loss function and optimizer.
        
        Uses binary cross‑entropy loss (suitable for a single sigmoid output)
        and the Adam optimizer with a default learning rate.
        """
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, x_batch, y_batch):
        """
        Train the model for one batch of data.
        
        Args:
            x_batch: Batch of input sequences (shape: [batch_size, max_length]).
            y_batch: Batch of corresponding labels (shape: [batch_size, 1]).
        
        Returns:
            float: The loss value for the given batch.
        """
        history = self.model.train_on_batch(x_batch, y_batch)
        # train_on_batch returns [loss, *metrics]; we only need the loss
        return history[0]
    
    def predict(self, x_batch):
        """
        Generate sentiment predictions for a batch of input sequences.
        
        Args:
            x_batch: Batch of input sequences (shape: [batch_size, max_length]).
        
        Returns:
            np.ndarray: Array of predicted sentiment scores (shape: [batch_size, 1]).
        """
        return self.model.predict(x_batch, verbose=0)
    
    def save(self, path):
        """
        Save the entire model (architecture, weights, optimizer state) to disk.
        
        Uses the TensorFlow 2.x SavedModel format (default). The saved model can
        be reloaded with `tf.keras.models.load_model`.
        
        Args:
            path: Directory where the model will be saved.
        """
        self.model.save(path)
    
    def load(self, path):
        """
        Load a previously saved model from disk.
        
        Args:
            path: Directory containing the saved model.
        """
        self.model = tf.keras.models.load_model(path)


if __name__ == '__main__':
    # Example usage and quick sanity check
    model = SentimentLegacyModel()
    print("Modernized sentiment model initialized successfully.")
    print(f"Model summary:\n")
    model.model.summary()
    
    # Create dummy data to verify forward pass
    dummy_input = np.random.randint(0, 1000, size=(5, 200))
    dummy_pred = model.predict(dummy_input)
    print(f"\nDummy prediction shape: {dummy_pred.shape}")
    print(f"Sample predictions: {dummy_pred.flatten()[:3]}")