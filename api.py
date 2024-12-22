import os
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Input
from sklearn.preprocessing import LabelEncoder
import pickle


def parse_text_data_from_xml(xml_path):
    """
    Parses text data from an XML file.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        list: A list of tuples (text, label).
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    text_data = []

    for text_element in root.findall("text"):
        content = text_element.find("content").text
        label = text_element.find("label").text
        text_data.append((content, label))

    return text_data


def create_image_model(input_shape):
    """
    Creates an image model.

    Args:
        input_shape (tuple): Shape of the input image.

    Returns:
        tuple: (Input layer, Output layer).
    """
    image_input = Input(shape=input_shape, name="image_input")
    x = Conv2D(32, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(128, activation="relu")(x)
    return image_input, image_output


def create_text_model(max_sequence_length, vocab_size):
    """
    Creates a text model.

    Args:
        max_sequence_length (int): Maximum sequence length for text inputs.
        vocab_size (int): Vocabulary size.

    Returns:
        tuple: (Input layer, Output layer).
    """
    text_input = Input(shape=(max_sequence_length,), name="text_input")
    x = Embedding(vocab_size, 128, input_length=max_sequence_length)(text_input)
    x = LSTM(128)(x)
    text_output = Dense(128, activation="relu")(x)
    return text_input, text_output


def train_text_model(text_data, output_path, max_sequence_length=None):
    """
    Trains a text classification model.

    Args:
        text_data (list): A list of tuples (text, label).
        output_path (str): Path to save the trained model.
        max_sequence_length (int, optional): Maximum sequence length for text inputs. If None, it is computed from the data.
    """
    texts, labels = zip(*text_data)

    # Tokenize the texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    if not sequences:
        raise ValueError("No sequences generated. Check if text data is valid.")

    # Compute or set the max sequence length
    if max_sequence_length is None:
        max_sequence_length = max(len(seq) for seq in sequences)

    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
    vocab_size = len(tokenizer.word_index) + 1

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into training and validation sets
    split_index = int(0.8 * len(padded_sequences))
    train_sequences, validation_sequences = padded_sequences[:split_index], padded_sequences[split_index:]
    train_labels, validation_labels = encoded_labels[:split_index], encoded_labels[split_index:]

    # Create the text model
    text_input, text_output = create_text_model(max_sequence_length, vocab_size)
    combined_output = Dense(len(label_encoder.classes_), activation="softmax")(text_output)
    model = Model(inputs=text_input, outputs=combined_output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(
        np.array(train_sequences),
        np.array(train_labels),
        epochs=10,
        validation_data=(np.array(validation_sequences), np.array(validation_labels)),
    )

    # Save the model, tokenizer, and label encoder
    model.save(output_path)
    print(f"Text model saved to {output_path}")

    with open(output_path + "_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open(output_path + "_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)


def train_image_model(image_folder, output_path, mode):
    """
    Trains an image model for classification or generation.

    Args:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path to save the trained model.
        mode (int): 2 for image generation, 3 for image classification.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        image_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        image_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    input_shape = (150, 150, 3)
    image_input, image_output = create_image_model(input_shape)

    if mode == 2:  # Image Generation (example dummy model)
        combined_output = Dense(128, activation="relu")(image_output)
    elif mode == 3:  # Image Classification
        combined_output = Dense(len(train_gen.class_indices), activation="softmax")(image_output)

    model = Model(inputs=image_input, outputs=combined_output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
    )

    model.save(output_path)
    print(f"Image model saved to {output_path}")


def main(xml_path, image_folder, output_path, mode, max_sequence_length=None):
    """
    Main function for training models.

    Args:
        xml_path (str): Path to the XML file containing text data.
        image_folder (str): Path to the folder containing image data.
        output_path (str): Path to save the trained model.
        mode (int): Mode of operation.
                    1: Train text model.
                    2: Train image generation model.
                    3: Train image classification model.
        max_sequence_length (int, optional): Maximum sequence length for text inputs. Defaults to None.
    """
    if mode == 1:  # Train text model
        if not xml_path:
            raise ValueError("XML path is required for text model training.")
        text_data = parse_text_data_from_xml(xml_path)
        if not text_data:
            raise ValueError("No text data available in the XML file.")
        train_text_model(text_data, output_path, max_sequence_length)
    elif mode in [2, 3]:  # Train image models
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        train_image_model(image_folder, output_path, mode)
    else:
        raise ValueError("Invalid mode. Choose 1 (Text Model), 2 (Image Generation), or 3 (Image Classification).")


# Example usage:
# Train text model using XML data with default max sequence length
# main("training_data.xml", None, "text_model.h5", 1)

# Train text model with a custom max sequence length
# main("training_data.xml", None, "text_model.h5", 1, max_sequence_length=100)

# Train image classification model using image directory
# main(None, "path/to/image_folder", "image_model.h5", 3)
