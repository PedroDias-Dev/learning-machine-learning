# IMPORTANT: this algorithm is mostly a test for text based data and for model loading, and is not optimized for real world use cases.

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import csv

# Load the dataset from a CSV file
def load_dataset(dataset_file):
    plaintexts = []
    encrypted_texts = []
    with open(dataset_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            plaintexts.append(row[0])
            encrypted_texts.append(row[1])
    return plaintexts, encrypted_texts

plaintexts, encrypted_texts = load_dataset('encryption_dataset.csv')

# Define the vocabulary and mappings
vocab = sorted(set(''.join(plaintexts + encrypted_texts)))
vocab_size = len(vocab)

char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = {index: char for index, char in enumerate(vocab)}

# Convert plaintext and encrypted_text to numerical sequences
plaintexts_numerical = [[char_to_index[char] for char in text] for text in plaintexts]
encrypted_texts_numerical = [[char_to_index[char] for char in text] for text in encrypted_texts]

# Pad sequences to a fixed length
max_seq_length = max(len(text) for text in plaintexts_numerical)
plaintexts_numerical = tf.keras.preprocessing.sequence.pad_sequences(plaintexts_numerical, maxlen=max_seq_length, padding='post')
encrypted_texts_numerical = tf.keras.preprocessing.sequence.pad_sequences(encrypted_texts_numerical, maxlen=max_seq_length, padding='post')

# Check if model exists, and if it does, loads it
try:
    model = keras.models.load_model('./models/encryption_model.keras')
    print('Model loaded successfully')
except OSError:
    print('Model not found, creating a new one')
    # Define the sequence-to-sequence model
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 32, input_length=max_seq_length),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dense(vocab_size, activation='softmax')
    ])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(plaintexts_numerical, encrypted_texts_numerical, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the model's performance
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')

# Use the model for decryption by providing plaintext input
plaintext_to_predict = ['HELLO', 'WORLD', 'MACHINE', 'LEARNING']
plaintexts_numerical_to_predict = [[char_to_index[char] for char in text] for text in plaintext_to_predict]
plaintexts_numerical_to_predict = tf.keras.preprocessing.sequence.pad_sequences(plaintexts_numerical_to_predict, maxlen=max_seq_length, padding='post')
predictions_numerical = model.predict(plaintexts_numerical_to_predict)
predictions = [''.join([index_to_char[np.argmax(prob)] for prob in prediction]) for prediction in predictions_numerical]

for i in range(len(plaintext_to_predict)):
    print(f'Original: {plaintext_to_predict[i]}, Decrypted Prediction: {predictions[i]}')

model.save('./models/encryption_model.keras')
