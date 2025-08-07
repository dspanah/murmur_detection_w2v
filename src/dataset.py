import pandas as pd
from datasets import load_dataset
import os


def append_path_to_df(df_path, audio_path, column):
    """
    Appends a base audio path to the values in a specific column of a CSV file.

    Parameters:
        df_path (str): Path to the CSV file.
        audio_path (str): Base path to prepend to file names in the column.
        column (str): Name of the column whose values will be updated.

    Returns:
        pd.DataFrame: Updated DataFrame with modified column values.
    """
    df = pd.read_csv(df_path)  # Load the CSV into a DataFrame
    for i in range(len(df)):
        # Prepend audio_path to each entry in the specified column
        df.loc[i, [column]] = audio_path + df[column][i]

    return df


def shuffle_df(df):
    """
    Shuffles the rows of the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Shuffled DataFrame.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle while keeping reproducibility
    return df


def save_df(df, path):
    """
    Saves a DataFrame to a CSV file without the index.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        path (str): Output file path.
    """
    df.to_csv(path, index=False)


def prepare_dataset(train_path, valid_path, audio_path, input_column, target_column):
    """
    Prepares a training and validation dataset from CSV files by:
    - Appending the full path to audio file names
    - Shuffling the data
    - Saving the modified files
    - Loading them using HuggingFace `load_dataset`

    Parameters:
        train_path (str): Path to the training CSV.
        valid_path (str): Path to the validation CSV.
        audio_path (str): Base path to prepend to input_column values.
        input_column (str): Column name containing file names.
        target_column (str): Column name containing labels.

    Returns:
        Tuple: (train_dataset, eval_dataset, labels)
            - train_dataset: HuggingFace dataset for training.
            - eval_dataset: HuggingFace dataset for validation.
            - labels: Sorted list of unique labels in the training dataset.
    """
    
    os.makedirs("tmp", exist_ok=True)
    
    # Add audio path to input column in both datasets
    df_train = append_path_to_df(train_path, audio_path, input_column)
    df_eval = append_path_to_df(valid_path, audio_path, input_column)

    # Shuffle both datasets
    df_train = shuffle_df(df_train)
    df_eval = shuffle_df(df_eval)

    # Create new file paths for the modified datasets
    train_path = "./tmp/train_set_incl_path.csv"
    valid_path = "./tmp/dev_set_incl_path.csv"

    # Save the modified datasets
    save_df(df_train, train_path)
    save_df(df_eval, valid_path)

    # Load the datasets with HuggingFace
    data_files = {
        "train": train_path,
        "validation": valid_path,
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Get sorted list of unique labels from training data
    labels = train_dataset.unique(target_column)
    labels.sort()

    return train_dataset, eval_dataset, labels


def prepare_test_dataset(test_path, audio_path, input_column, target_column):
    """
    Prepares a test dataset similarly to the train/validation datasets by:
    - Appending full path to audio files
    - Saving and loading the updated dataset
    - Extracting the list of labels

    Parameters:
        test_path (str): Path to the test CSV.
        audio_path (str): Base path to prepend to input_column values.
        input_column (str): Column name with file names.
        target_column (str): Column name with ground truth labels.

    Returns:
        Tuple: (test_dataset, labels)
            - test_dataset: HuggingFace dataset for testing.
            - labels: Sorted list of unique labels in the test dataset.
    """
    # Modify audio path in the input column
    df_test = append_path_to_df(test_path, audio_path, input_column)

    # Define new filename for updated test set
    df_path = "./tmp/test_set_incl_path.csv"

    # Save the updated test set
    save_df(df_test, df_path)

    # Load it using HuggingFace
    data_files = {
        "test": df_path,
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter=",")

    test_dataset = dataset["test"]

    # Extract and sort unique labels
    labels = test_dataset.unique(target_column)
    labels.sort()

    return test_dataset, labels