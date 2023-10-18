import random
from crypt import encrypt

# Parameters
num_samples = 100  # Number of samples to generate
max_text_length = 20  # Maximum text length

# Generate and store the dataset
dataset = []

for _ in range(num_samples):
    plaintext = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 ') for _ in range(random.randint(1, max_text_length)))
    key = random.randint(1, 26)  # Vary the encryption key
    encrypted_text = encrypt(plaintext, key)
    dataset.append((plaintext, encrypted_text))

# Save the dataset to a CSV file
import csv

with open('encryption_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['plaintext', 'encrypted_text'])
    for plaintext, encrypted_text in dataset:
        writer.writerow([plaintext, encrypted_text])
