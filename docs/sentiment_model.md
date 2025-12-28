# Sentiment Analysis Model Documentation

## Overview

This sentiment analysis model is a neural network designed to classify the sentiment of text as either positive or negative. It processes integer‑encoded text sequences and outputs a probability score between 0 (negative) and 1 (positive). The model is built with TensorFlow 2.x and the Keras API, making it easy to train, evaluate, and deploy.

## Architecture

The model follows a straightforward feed‑forward architecture that combines an embedding layer with two fully‑connected hidden layers.

### Layer‑by‑Layer Description

1. **Input Layer**  
   - Shape: `(max_length,)` where `max_length` is the maximum allowed sequence length (default 200).  
   - Each input is a sequence of integer token IDs.

2. **Embedding Layer**  
   - `input_dim`: Size of the vocabulary (`vocab_size`, default 10,000).  
   - `output_dim`: Dimension of the dense embedding vectors (`embedding_dim`, default 128).  
   - `input_length`: Fixed sequence length (`max_length`).  
   - This layer learns a dense representation for each token in the vocabulary. The embedding matrix is trained together with the rest of the network.

3. **Flatten Layer**  
   - Converts the 2‑D tensor of shape `(batch_size, max_length, embedding_dim)` into a 1‑D tensor of size `max_length * embedding_dim`. This step is necessary because the subsequent dense layers expect a flat feature vector.

4. **Dense (Fully‑Connected) Layers**  
   - **First dense layer**: 256 units with ReLU activation.  
   - **Second dense layer**: 128 units with ReLU activation.  
   - These hidden layers capture non‑linear interactions among the embedded features.

5. **Output Layer**  
   - Single neuron with a sigmoid activation function.  
   - Produces a scalar value between 0 and 1 representing the probability of positive sentiment.

### Diagram (Textual)

```
Input (max_length integers)
      ↓
Embedding (vocab_size × embedding_dim)
      ↓
Flatten (max_length * embedding_dim features)
      ↓
Dense(256, relu)
      ↓
Dense(128, relu)
      ↓
Dense(1, sigmoid) → Sentiment score
```

## Embedding Technique

The model uses a **trainable embedding layer** that is initialized randomly and updated during training. This is a form of **word embedding** where each token (word or sub‑word) is mapped to a dense, low‑dimensional vector. The embeddings are learned end‑to‑end together with the classification task, allowing the model to discover semantic relationships that are useful for sentiment analysis.

If pre‑trained embeddings (e.g., Word2Vec, GloVe, FastText) are available, they can be loaded into the embedding matrix as an initialization step, potentially improving performance with limited labeled data.

## Usage Example

Below is a simple example that shows how to instantiate the model, make a prediction, and save/load the trained weights.

### Import the Model

```python
from src.models.sentiment_legacy import SentimentLegacyModel
```

### Instantiate the Model

```python
model = SentimentLegacyModel(
    vocab_size=10000,
    embedding_dim=128,
    max_length=200
)
```

### Inspect the Model Architecture

```python
model.model.summary()
```

### Make a Prediction

Assume you have already tokenized your text and converted it to integer sequences of length `max_length` (padding or truncating as needed). The input shape should be `(batch_size, max_length)`.

```python
import numpy as np

# Example batch of 3 sequences (each of length 200)
sample_input = np.random.randint(0, 10000, size=(3, 200))

# Get sentiment scores
predictions = model.predict(sample_input)
print(predictions)  # shape (3, 1)
```

### Train the Model

The model can be trained using the `train` method for batch‑wise updates, or you can directly call `model.model.fit()` for full‑dataset training.

```python
# Dummy training data
x_train = np.random.randint(0, 10000, size=(32, 200))
y_train = np.random.rand(32, 1)  # random labels between 0 and 1

# Train for one batch
loss = model.train(x_train, y_train)
print(f"Batch loss: {loss}")
```

### Save and Load the Model

```python
# Save the entire model (architecture + weights + optimizer state)
model.save('path/to/saved_model')

# Later, load the saved model
loaded_model = SentimentLegacyModel()
loaded_model.load('path/to/saved_model')
```

## Notes and Best Practices

- **Vocabulary Size**: Ensure `vocab_size` matches the number of unique tokens in your tokenizer. Tokens not in the vocabulary will be mapped to index 0 (the padding index).
- **Sequence Length**: All input sequences must be padded/truncated to exactly `max_length`. Consider using a tokenizer that handles padding automatically (e.g., `tf.keras.preprocessing.sequence.pad_sequences`).
- **Embedding Dimension**: Larger embedding dimensions can capture more fine‑grained semantics but increase the number of parameters and risk overfitting.
- **Training**: For best results, train the model on a large, balanced dataset of labeled text. You may also fine‑tune the embeddings using pre‑trained vectors.
- **Deployment**: The model can be exported as a SavedModel and served with TensorFlow Serving or integrated into a Python web application.

## References

- [TensorFlow 2.x Documentation](https://www.tensorflow.org/guide/keras)
- [Keras Embedding Layer](https://keras.io/api/layers/core_layers/embedding/)
- [Word Embeddings: A Survey](https://arxiv.org/abs/1901.09069)