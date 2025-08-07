import pandas as pd
import numpy as np
import math
import yaml

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def id_extractor(filename):
    """
    Extracts the patient ID and recording ID from a filename.

    Parameters:
        filename (str): The input filename.

    Returns:
        tuple: (recording_id, patient_id)
    """
    fname_lst = filename.split("_")
    patient_id = fname_lst[0]
    recording_id = fname_lst[0] + "_" + fname_lst[1]  # patientID_recordingID
    return recording_id, patient_id


def argmax(lst):
    """
    Returns the index of the maximum value in a list.

    Parameters:
        lst (list): Input list of values.

    Returns:
        int: Index of the max value.
    """
    return max(range(len(lst)), key=lst.__getitem__)


def preds_agg(pred_lst):
    """
    Aggregates multiple predictions into a single class using a priority rule.

    Priority: If any segment is predicted as 1 (Present), return 1.
              Else if any is 2 (Unknown), return 2.
              Else return 0 (Absent).

    Parameters:
        pred_lst (pd.Series): List or Series of integer predictions.

    Returns:
        int: Aggregated prediction.
    """
    if 1 in pred_lst.to_list():
        pred = 1
    elif 2 in pred_lst.to_list():
        pred = 2
    else:
        pred = 0
    return pred


def get_patient_result(labels, preds):
    """
    Aggregates segment-level predictions to get patient-level predictions.

    Steps:
    1. Create a DataFrame with segment-level labels and predictions.
    2. Extract metadata (filename, recording ID, patient ID).
    3. Aggregate predictions per recording using argmax.
    4. Aggregate predictions per patient using preds_agg.
    5. Save intermediate DataFrames and return final labels and predictions.

    Parameters:
        labels (list[int]): Ground-truth labels for each sample.
        preds (list[list[int]]): List of prediction scores for classes [0, 1, 2] for each sample.

    Returns:
        tuple: (labels, preds) at the **patient level**.
    """

    # Build initial DataFrame
    df = pd.DataFrame()
    df["label"] = labels
    df["pred_0"] = [item[0] for item in preds]  # Probability/score for class 0
    df["pred_1"] = [item[1] for item in preds]  # Probability/score for class 1
    df["pred_2"] = [item[2] for item in preds]  # Probability/score for class 2

    # Load filenames from test set CSV and add to DataFrame
    config = load_config("config.yaml")
    TEST_PATH = config["data"]["test_path"]
    df_test_set = pd.read_csv(TEST_PATH)
    fname_lst = df_test_set["Filename"].to_list()
    df["filename"] = fname_lst

    # Extract recording ID and patient ID for each filename
    recording_ids = []
    patient_ids = []

    for i in range(len(df)):
        filename = df.iloc[i]["filename"]
        recording_id, patient_id = id_extractor(filename)
        recording_ids.append(recording_id)
        patient_ids.append(patient_id)

    df["recording_id"] = recording_ids
    df["patient_id"] = patient_ids

    # Save segment-level DataFrame
    df.to_csv("./tmp/df_segment.csv", index=False)

    # Group by recording_id to aggregate segment-level predictions
    df_recording = df.groupby(['recording_id'], as_index=False).agg({
        'label': 'first',           # Assuming all segments of a recording have the same label
        'pred_0': 'mean',           # Average prediction scores across segments
        'pred_1': 'mean',
        'pred_2': 'mean',
        'recording_id': 'first',
        'patient_id': 'first'
    })

    # Derive final prediction per recording based on max of average prediction scores
    preds_lst = []
    for i in range(len(df_recording)):
        preds = [
            df_recording.iloc[i]["pred_0"],
            df_recording.iloc[i]["pred_1"],
            df_recording.iloc[i]["pred_2"]
        ]
        pred_max_ix = argmax(preds)
        preds_lst.append(pred_max_ix)

    df_recording["pred"] = preds_lst

    # Drop the raw class score columns
    df_recording.drop(labels=["pred_0", "pred_1", "pred_2"], axis=1, inplace=True)

    # Save recording-level predictions
    df_recording.to_csv("./tmp/df_recording.csv", index=False)

    # Group by patient to aggregate multiple recordings
    df_patient = df_recording.groupby(['patient_id'], as_index=False).agg({
        'patient_id': 'first',
        'label': 'first',      # Assuming each patient has one consistent label
        'pred': preds_agg      # Use aggregation rule for final prediction
    })

    # Save patient-level predictions
    df_patient.to_csv("./tmp/df_patient.csv", index=False)

    # Extract final label and prediction lists
    labels = df_patient["label"].to_list()
    preds = df_patient["pred"].to_list()

    return labels, preds
