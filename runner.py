import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_tokenizer(tokenizer_path):
    """
    Loads a saved tokenizer.

    Args:
        tokenizer_path (str): Path to the saved tokenizer.

    Returns:
        Tokenizer: The loaded tokenizer.
    """
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def load_label_encoder(label_encoder_path):
    """
    Loads a saved label encoder.

    Args:
        label_encoder_path (str): Path to the saved label encoder.

    Returns:
        LabelEncoder: The loaded label encoder.
    """
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder


def run_text_model(model_path, input_text, tokenizer, max_sequence_length, label_encoder):
    """
    Runs a text model for predictions and returns the most probable label.

    Args:
        model_path (str): Path to the saved model.
        input_text (str): The input text to be classified or processed.
        tokenizer (Tokenizer): Tokenizer used during training.
        max_sequence_length (int): Maximum sequence length used during training.
        label_encoder (LabelEncoder): LabelEncoder used during training.

    Returns:
        tuple: (probabilities, most_probable_label)
            - probabilities (numpy.ndarray): Prediction probabilities for each class.
            - most_probable_label (str): The label with the highest probability.
    """
    model = load_model(model_path)
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
    probabilities = model.predict(padded_sequence)
    most_probable_index = np.argmax(probabilities)
    most_probable_label = label_encoder.inverse_transform([most_probable_index])[0]
    return probabilities[0], most_probable_label


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


def main(model_path, mode, tokenizer=None, max_sequence_length=None, input_text=None, image_path=None, label_encoder=None):
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
        label_encoder (LabelEncoder, optional): Label encoder used for text models. Required for mode 1.

    Returns:
        None
    """
    if mode == 1:  # Text model
        if not tokenizer or not max_sequence_length or not input_text or not label_encoder:
            raise ValueError("For text mode, tokenizer, max_sequence_length, input_text, and label_encoder are required.")
        probabilities, most_probable_label = run_text_model(
            model_path, input_text, tokenizer, max_sequence_length, label_encoder
        )
        
        return most_probable_label
        #return probabilities
    elif mode in [2, 3]:  # Image models
        if not image_path:
            raise ValueError("For image mode, image_path is required.")
        prediction = run_image_model(model_path, image_path)
    else:
        raise ValueError("Invalid mode. Choose 1 (Text Model), 2 (Image Generation), or 3 (Image Classification).")
