import runner

tokenizer =runner.Tokenizer()

runner.main(
    model_path="text_model.h5",
    mode=1,
    tokenizer=tokenizer,
    max_sequence_length=100,
    input_text="The goverment",
)