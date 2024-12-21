import os
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, Input


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
    image_input = Input(shape=input_shape, name="image_input")
    x = Conv2D(32, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(128, activation="relu")(x)
    return image_input, image_output


def create_text_model(max_sequence_length, vocab_size):
    text_input = Input(shape=(max_sequence_length,), name="text_input")
    x = Embedding(vocab_size, 128, input_length=max_sequence_length)(text_input)
    x = LSTM(128)(x)
    text_output = Dense(128, activation="relu")(x)
    return text_input, text_output


def train_text_model(text_data, output_path):
    texts, labels = zip(*text_data)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    if not sequences:
        raise ValueError("No sequences generated. Check if text data is valid.")

    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
    vocab_size = len(tokenizer.word_index) + 1

    split_index = int(0.8 * len(padded_sequences))
    train_sequences, validation_sequences = padded_sequences[:split_index], padded_sequences[split_index:]
    train_labels, validation_labels = labels[:split_index], labels[split_index:]

    text_input, text_output = create_text_model(max_sequence_length, vocab_size)
    combined_output = Dense(len(set(labels)), activation="softmax")(text_output)
    model = Model(inputs=text_input, outputs=combined_output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        np.array(train_sequences),
        np.array(train_labels),
        epochs=10,
        validation_data=(np.array(validation_sequences), np.array(validation_labels)),
    )

    model.save(output_path)
    print(f"Text model saved to {output_path}")


def train_image_model(image_folder, output_path, mode):
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


def main(xml_path, image_folder, output_path, mode):
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
    """
    if mode == 1:  # Train text model
        text_data = parse_text_data_from_xml(xml_path)
        if not text_data:
            raise ValueError("No text data available in the XML file.")
        train_text_model(text_data, output_path)
    elif mode in [2, 3]:  # Train image models
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        train_image_model(image_folder, output_path, mode)
    else:
        raise ValueError("Invalid mode. Choose 1 (Text Model), 2 (Image Generation), or 3 (Image Classification).")


# Example usage:
# main("training_data.xml", "images", "output_text_model.h5", 1)  # Train text model
# main(None, "images", "output_image_model.h5", 3)  # Train image classification model
