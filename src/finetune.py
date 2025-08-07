import torch
import torchaudio
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import AutoFeatureExtractor
from dataset import prepare_dataset
from torch import nn
from custom_metric import weighted_accuracy
from patient_result import get_patient_result
from transformers import EarlyStoppingCallback
from seed import set_global_seed
import yaml

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class DataPreprocess:
    """
    Handles audio loading, label mapping, and feature extraction for audio classification.

    Args:
        input_column (str): Name of the column containing audio file paths.
        target_column (str): Name of the column containing labels.
        labels (List[str]): List of possible label strings.
        pretrained_model_path (str): Path to the pretrained model.
        target_sampling_rate (int): Desired sampling rate for audio processing.
        hs_min_len (int): Minimum heart sound length threshold.
    """
    def __init__(self, input_column, target_column, labels, pretrained_model_path, target_sampling_rate, hs_min_len):
        self.input_column = input_column
        self.target_column = target_column
        self.labels = labels
        self.pretrained_model_path = pretrained_model_path
        self.target_sampling_rate = target_sampling_rate
        self.hs_min_len = hs_min_len

    def label_id_map(self):
        """
        Generate mappings from label to ID and ID to label.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: label2id and id2label mappings.
        """
        label2id = {label: str(i) for i, label in enumerate(self.labels)}
        id2label = {str(i): label for i, label in enumerate(self.labels)}
        return label2id, id2label

    @staticmethod
    def load_audio(audio_path):
        """
        Load and return audio as a numpy array.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: Loaded audio signal.
        """
        audio, sr = torchaudio.load(audio_path)
        return audio.squeeze().numpy()

    @staticmethod
    def label_to_id(label, label_list):
        """
        Convert label to its corresponding ID.

        Args:
            label (str): The label to convert.
            label_list (List[str]): List of valid labels.

        Returns:
            int: Corresponding ID, or -1 if label is not found.
        """
        return label_list.index(label) if label in label_list else -1 if label_list else label

    def make_feature_extractor(self):
        """
        Load feature extractor from pretrained model.

        Returns:
            AutoFeatureExtractor: Hugging Face feature extractor.
        """
        return AutoFeatureExtractor.from_pretrained(self.pretrained_model_path)

    def preprocess(self, examples):
        """
        Process a batch of examples into model-ready inputs.

        Args:
            examples (Dict[str, List]): Dictionary with audio paths and labels.

        Returns:
            Dict[str, List]: Processed features and labels.
        """
        audio_list = [self.load_audio(path) for path in examples[self.input_column]]
        target_list = [self.label_to_id(label, self.labels) for label in examples[self.target_column]]
        feature_extractor = self.make_feature_extractor()
        result = feature_extractor(audio_list, sampling_rate=self.target_sampling_rate, batched=True)
        result["label"] = target_list
        return result

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for classification.

    Args:
        eval_pred (EvalPrediction): Predictions and labels.

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    accuracy_metric = evaluate.load("accuracy")

    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)

    results = {
        **accuracy_metric.compute(references=labels, predictions=preds),
        **f1_metric.compute(references=labels, predictions=preds, average="macro"),
        **recall_metric.compute(references=labels, predictions=preds, average="macro"),
        **precision_metric.compute(references=labels, predictions=preds, average="macro"),
        **weighted_accuracy(labels=labels, outputs=preds),
        'recall_absent': recall[0],
        'recall_present': recall[1],
        'recall_unknown': recall[2],
    }
    return results

def load_pretrained_model(model_path, dp):
    """
    Load a pretrained audio classification model and apply label mappings.

    Args:
        model_path (str): Path to pretrained model.
        dp (DataPreprocess): DataPreprocess instance.

    Returns:
        PreTrainedModel: Model ready for fine-tuning.
    """
    label2id, id2label = dp.label_id_map()
    num_labels = len(id2label)
    #print("lebel_0", id2label[str(0)])
    #print("lebel_1", id2label[str(1)])
    #print("lebel_2", id2label[str(2)])
    return AutoModelForAudioClassification.from_pretrained(
        model_path, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

def freeze_feature_encoder(model):
    """
    Freeze the feature encoder of the model.

    Args:
        model (PreTrainedModel): Audio model.

    Returns:
        PreTrainedModel: Model with frozen feature encoder.
    """
    model.freeze_feature_encoder()
    return model

def freeze_base_model(model):
    """
    Freeze the base model.

    Args:
        model (PreTrainedModel): Audio model.

    Returns:
        PreTrainedModel: Model with frozen base layers.
    """
    model.freeze_base_model()
    return model

class CustomTrainer(Trainer):
    """
    Trainer with custom weighted loss function for imbalanced class labels.
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute weighted cross-entropy loss.

        Args:
            model (PreTrainedModel): Model instance.
            inputs (Dict[str, torch.Tensor]): Input batch.
            return_outputs (bool): Whether to return outputs along with loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: Loss (and outputs if requested).
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = torch.device('cuda:0')
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 5.5, 20.5]).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":

    config = load_config("config.yaml")

    # Configurations
    INPUT_COLUMN = "Filename"
    TARGET_COLUMN = "Murmur"
    TARGET_SR = 16000
    HS_MIN_LEN = 5
    set_global_seed(42)
    
    TRAIN_PATH = config["data"]["train_path"]
    VALID_PATH = config["data"]["valid_path"]
    TEST_PATH = config["data"]["test_path"]
    TRAIN_AUDIO_PATH = config["data"]["audio_path"]
    TEST_AUDIO_PATH = config["data"]["audio_path"]
    PT_MODEL_PATH = config["model"]["pt_model_path"]

    # Load and prepare datasets
    train_dataset, eval_dataset, LABELS = prepare_dataset(TRAIN_PATH, VALID_PATH, TRAIN_AUDIO_PATH, INPUT_COLUMN, TARGET_COLUMN)
    dataprocessor = DataPreprocess(INPUT_COLUMN, TARGET_COLUMN, LABELS, PT_MODEL_PATH, TARGET_SR, HS_MIN_LEN)

    train_set = train_dataset.map(dataprocessor.preprocess, batched=True, remove_columns=[INPUT_COLUMN, TARGET_COLUMN, "Outcome"])
    eval_set = eval_dataset.map(dataprocessor.preprocess, batched=True, remove_columns=[INPUT_COLUMN, TARGET_COLUMN, "Outcome"])

    # Load model and tokenizer
    pretrained_model = load_pretrained_model(PT_MODEL_PATH, dataprocessor)
    tokenizer = dataprocessor.make_feature_extractor()

    # Training parameters
    batch_size = 8
    epochs = 20
    lr = 3e-5
    total_steps = int((np.ceil(train_set.num_rows / batch_size) * epochs))

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=total_steps, epochs=epochs)

    early_stop = EarlyStoppingCallback(5)

    training_args = TrainingArguments(
        output_dir="ft_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        seed=42
    )

    # Initialize and start training
    trainer = CustomTrainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[early_stop]
    )

    trainer.train()
