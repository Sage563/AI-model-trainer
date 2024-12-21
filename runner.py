import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def run_text_model(model_path, input_text, tokenizer, max_sequence_length):
    """
    Runs a text model for predictions.

    Args:
        model_path (str): Path to the saved model.
        input_text (str): The input text to be classified or processed.
        tokenizer (Tokenizer): Tokenizer used during training.
        max_sequence_length (int): Maximum sequence length used during training.

    Returns:
        numpy.ndarray: Prediction from the model.
    """
    model = load_model(model_path)
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded_sequence)
    return prediction


def run_image_model(model_path, image_path, target_size=(150, 150)):
    """
    Runs an image model for predictions.

    Args:
        model_path (str): Path to the saved model.
        image_path (str): Path to the image to be classified or processed.
        target_size (tuple): Target size for resizing the image.

    Returns:
        numpy.ndarray: Prediction from the model.
    """
    model = load_model(model_path)
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return prediction


def main(model_path, mode, tokenizer=None, max_sequence_length=None, input_text=None, image_path=None):
    """
    Main function to run a trained model.

    Args:
        model_path (str): Path to the saved model.
        mode (int): Mode of operation.
                    1: Text model.
                    2: Image generation model.
                    3: Image classification model.
        tokenizer (Tokenizer, optional): Tokenizer used for text models. Required for mode 1.
        max_sequence_length (int, optional): Maximum sequence length for text models. Required for mode 1.
        input_text (str, optional): Input text for text models. Required for mode 1.
        image_path (str, optional): Path to the image. Required for mode 2 and 3.

    Returns:
        None
    """
    if mode == 1:  # Text model
        if not tokenizer or not max_sequence_length or not input_text:
            raise ValueError("For text mode, tokenizer, max_sequence_length, and input_text are required.")
        prediction = run_text_model(model_path, input_text, tokenizer, max_sequence_length)
    elif mode in [2, 3]:  # Image models
        if not image_path:
            raise ValueError("For image mode, image_path is required.")
        prediction = run_image_model(model_path, image_path)
    else:
        raise ValueError("Invalid mode. Choose 1 (Text Model), 2 (Image Generation), or 3 (Image Classification).")

    print("Prediction:", prediction)


# Example Usage
# For text model:
# tokenizer = Tokenizer()  # Load or recreate the tokenizer used during training
# main(
#     model_path="text_model.h5",
#     mode=1,
#     tokenizer=tokenizer,
#     max_sequence_length=100,
#     input_text="Sample input text"
# )

# For image model:
# main(
#     model_path="image_model.h5",
#     mode=3,
#     image_path="path/to/image.jpg"
# )
