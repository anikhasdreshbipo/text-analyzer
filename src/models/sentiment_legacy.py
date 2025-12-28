"""
Legacy sentiment analysis model using TensorFlow 1.x patterns.
This model is outdated and difficult to maintain.
"""

import tensorflow as tf
import numpy as np

class SentimentLegacyModel:
    """A legacy sentiment analysis model using word embeddings and dense layers."""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def _build_model(self):
        """Build the model graph using TensorFlow 1.x style."""
        # Placeholders
        self.input_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_length], name='input')
        self.labels_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='labels')
        
        # Embedding layer
        with tf.device('/cpu:0'):
            self.embedding_matrix = tf.compat.v1.get_variable(
                'embedding_matrix',
                shape=[self.vocab_size, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-0.1, 0.1)
            )
            embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.input_ph)
        
        # Flatten and dense layers
        flattened = tf.reshape(embedded, [-1, self.max_length * self.embedding_dim])
        dense1 = tf.layers.dense(flattened, units=256, activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(dense1, units=128, activation=tf.nn.relu, name='dense2')
        self.logits = tf.layers.dense(dense2, units=1, activation=None, name='logits')
        self.predictions = tf.sigmoid(self.logits)
        
        # Loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels_ph, logits=self.logits
        ))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
    
    def train(self, x_batch, y_batch):
        """Train the model for one batch."""
        feed_dict = {
            self.input_ph: x_batch,
            self.labels_ph: y_batch
        }
        _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss_val
    
    def predict(self, x_batch):
        """Make predictions."""
        feed_dict = {self.input_ph: x_batch}
        preds = self.sess.run(self.predictions, feed_dict=feed_dict)
        return preds
    
    def save(self, path):
        """Save the model using TensorFlow 1.x Saver."""
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, path)
    
    def load(self, path):
        """Load the model."""
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, path)


if __name__ == '__main__':
    # Example usage
    model = SentimentLegacyModel()
    print("Legacy model initialized.")