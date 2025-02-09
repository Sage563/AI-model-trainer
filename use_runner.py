import runner

# Load tokenizer and label encoder
tokenizer = runner.load_tokenizer("text_model.h5_tokenizer.pkl")
label_encoder = runner.load_label_encoder("text_model.h5_label_encoder.pkl")

# Run the text model
answ = runner.main(
    model_path="text_model.h5",
    mode=1,
    tokenizer=tokenizer,
    max_sequence_length=100,
    input_text="sports",
    label_encoder=label_encoder
)


print(answ)