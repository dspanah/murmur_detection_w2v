from sklearn.metrics import precision_recall_fscore_support
from transformers import Trainer
from custom_metric import weighted_accuracy
from dataset import prepare_test_dataset
from finetune import DataPreprocess, load_pretrained_model
import evaluate
from patient_result import get_patient_result
import yaml

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_patient_metrics(eval_pred):
    """
    Computes classification metrics (F1, recall, precision, and weighted accuracy) 
    at the patient level.

    Args:
        eval_pred: An EvalPrediction object containing `predictions` and `label_ids`.

    Returns:
        dict: A dictionary containing calculated metrics, including class-wise recall.
    """
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")

    # Aggregate predictions and labels per patient
    labels, preds = get_patient_result(eval_pred.label_ids, eval_pred.predictions)

    # Calculate per-class precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)

    results = {}
    results.update(f1_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(weighted_accuracy(labels=labels, outputs=preds))

    # Add individual recall values for each class
    results.update({'recall_absent': recall[0]})
    results.update({'recall_present': recall[1]})
    results.update({'recall_unknown': recall[2]})

    return results


if __name__ == "__main__":
    
    config = load_config("config.yaml")

    # Configurations
    INPUT_COLUMN = "Filename"
    TARGET_COLUMN = "Murmur"  
    TARGET_SR = 16000
    HS_MIN_LEN = 5
    
    TEST_PATH = config["data"]["test_path"]
    AUDIO_PATH = config["data"]["audio_path"]
    FT_MODEL_PATH = config["model"]["ft_model_path"]

    # Load and prepare test dataset
    test_dataset, LABELS = prepare_test_dataset(
        TEST_PATH,
        AUDIO_PATH,
        INPUT_COLUMN,
        TARGET_COLUMN
    )

    # Initialize data processor
    dataprocessor = DataPreprocess(
        INPUT_COLUMN,
        TARGET_COLUMN,
        LABELS,
        FT_MODEL_PATH,
        TARGET_SR,
        HS_MIN_LEN
    )

    # Preprocess test set
    test_set = test_dataset.map(
        dataprocessor.preprocess,
        batched=True,
        remove_columns=[INPUT_COLUMN, TARGET_COLUMN, "Outcome"]
    )

    # Load model and tokenizer
    model = load_pretrained_model(FT_MODEL_PATH, dataprocessor)
    tokenizer = dataprocessor.make_feature_extractor()

    # Evaluate model on test set
    trainer = Trainer(
        model=model,
        compute_metrics=compute_patient_metrics,
    )

    # Run prediction and print results
    test_result = trainer.predict(test_set)
    print(test_result)