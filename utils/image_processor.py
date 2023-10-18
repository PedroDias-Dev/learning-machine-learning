from PIL import Image
import numpy as np


def process_handwritten_image(image_path):
    # Load your handwritten image
    handwritten_image = Image.open(image_path)

    # Convert the image to grayscale
    handwritten_image = handwritten_image.convert("L")

    # Resize the image to match the MNIST image size (28x28 pixels)
    handwritten_image = handwritten_image.resize((28, 28))

    # Convert the image to a NumPy array
    handwritten_data = np.array(handwritten_image)

    # Normalize the pixel values
    handwritten_data = handwritten_data / 255.0

    # Reshape the data to match the input shape expected by your model
    handwritten_data = handwritten_data.reshape(1, 28, 28)

    return handwritten_data
