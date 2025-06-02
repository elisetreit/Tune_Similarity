import pandas as pd
from src.data.preprocessing import clean_data, transform_data
from src.data.tokenizer import Tokenizer

def preprocess_abc_data(input_file, output_file):
    # Load the abc data
    df = pd.read_csv(input_file)

    # Clean the data
    df_cleaned = clean_data(df)

    # Transform the data
    df_transformed = transform_data(df_cleaned)

    # Tokenize the data
    tokenizer = Tokenizer()
    tokenized_data = tokenizer.tokenize(df_transformed)

    # Save the preprocessed data
    tokenized_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'data/abc_data.csv'  # Path to the input abc data
    output_file = 'data/preprocessed_data.csv'  # Path to save the preprocessed data
    preprocess_abc_data(input_file, output_file)