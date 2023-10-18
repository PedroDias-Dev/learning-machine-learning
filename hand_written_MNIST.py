# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Make predictions on a single image
predictions = model.predict(test_images)
sample_image_index = 0
predicted_label = predictions[sample_image_index].argmax()
print(f'Predicted label for the sample image: {predicted_label}')

# Plot the sample image
plt.figure()
plt.imshow(test_images[sample_image_index], cmap='gray')
plt.title(f'Actual Label: {test_labels[sample_image_index]}, Predicted Label: {predicted_label}')
plt.show()
