import os
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, concatenate, Input




def create_image_model(input_shape):
    image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_output = Dense(128, activation='relu')(x)
    return image_input, image_output


def create_text_model(max_sequence_length, vocab_size):
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    x = Embedding(vocab_size, 128, input_length=max_sequence_length)(text_input)
    x = LSTM(128)(x)
    text_output = Dense(128, activation='relu')(x)
    return text_input, text_output


def save_training_data_to_xml(image_filepaths, text_data, output_path):
    root = ET.Element("training_data")
    
    for filepath in image_filepaths:
        image_element = ET.SubElement(root, "image")
        filepath_element = ET.SubElement(image_element, "filepath")
        filepath_element.text = filepath
    
    for text, label in text_data:
        text_element = ET.SubElement(root, "text")
        text_content_element = ET.SubElement(text_element, "content")
        text_content_element.text = text
        text_label_element = ET.SubElement(text_element, "label")
        text_label_element.text = label
    
    tree = ET.ElementTree(root)
    tree.write(output_path)


def train_text_model(text_folder, output_path):
    texts, labels = [], []
    for label in os.listdir(text_folder):
        label_folder = os.path.join(text_folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                filepath = os.path.join(label_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        texts.append(content)
                        labels.append(label)

    if not texts:
        raise ValueError("No valid text data found in the specified folder.")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    if not sequences:
        raise ValueError("No sequences generated. Check if text data is valid.")

    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    split_index = int(0.8 * len(padded_sequences))
    train_sequences, validation_sequences = padded_sequences[:split_index], padded_sequences[split_index:]
    train_labels, validation_labels = labels[:split_index], labels[split_index:]

    text_input, text_output = create_text_model(max_sequence_length, vocab_size)
    combined_output = Dense(len(set(labels)), activation='softmax')(text_output)
    model = Model(inputs=text_input, outputs=combined_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        np.array(train_sequences),
        np.array(train_labels),
        epochs=10,
        validation_data=(np.array(validation_sequences), np.array(validation_labels))
    )

    model.save(output_path)
    print(f"Text model saved to {output_path}")


def train_image_model(image_folder, output_path, mode):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        image_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
    )
    val_gen = datagen.flow_from_directory(
        image_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
    )
    input_shape = (150, 150, 3)
    image_input, image_output = create_image_model(input_shape)

    if mode == 2:  # Image Generation (example dummy model)
        combined_output = Dense(128, activation='relu')(image_output)
    elif mode == 3:  # Image Classification
        combined_output = Dense(len(train_gen.class_indices), activation='softmax')(image_output)

    model = Model(inputs=image_input, outputs=combined_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen
    )

    model.save(output_path)
    print(f"Image model saved to {output_path}")


def main(output_path, mode ,fold):
    """
    Main function for training models.
    
    Args:
        output_path (str): Path to save the trained model.
        mode (int): Mode of operation.
                    1: Train text model.
                    2: Train image generation model.
                    3: Train image classification model.
    """
    folder = fold
    if os.path.isdir(folder):
        return folder
    if mode == 1:
        text_folder = os.path.join(folder, "texts")
        train_text_model(text_folder, output_path)
    elif mode in [2, 3]:
        image_folder = os.path.join(folder, "images")
        train_image_model(image_folder, output_path, mode)
    else:
        raise ValueError("Invalid mode. Choose 1 (Text Model), 2 (Image Generation), or 3 (Image Classification).")


# Example usage:
# main("output_model.h5", 1)  # Train text model and save to "output_model.h5"
# main("output_model_image.h5", 3)  # Train image classification model
