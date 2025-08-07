import numpy as np
from sklearn.metrics import confusion_matrix


def weighted_accuracy(labels, outputs):
    """
    Computes a custom weighted accuracy metric for multi-class classification,
    with class-specific importance defined via a weight matrix.

    Assumes three classes: ['Absent', 'Present', 'Unknown'].

    Parameters:
        labels (array-like): True class labels.
        outputs (array-like): Predicted class labels.

    Returns:
        dict: A dictionary with the key 'weighted_accuracy' and the computed value.
    """
    
    # Define a weight matrix indicating importance of correct classifications.
    # Each row corresponds to the predicted class, and each column to the true class.
    weights = np.array([[1, 5, 3],   # Prediction = Absent
                        [1, 5, 3],   # Prediction = Present
                        [1, 5, 3]])  # Prediction = Unknown

    # Compute the confusion matrix and transpose it so that:
    # Rows = predicted classes, Columns = true classes
    cf = confusion_matrix(labels, outputs).transpose()

    # Ensure the confusion matrix and weights have the same shape
    assert np.shape(cf) == np.shape(weights)

    # Element-wise multiply the confusion matrix by the weights
    wcf = weights * cf

    # Compute the weighted accuracy as the weighted correct predictions
    # divided by the total weighted predictions
    if np.sum(wcf) > 0:
        weighted_accuracy = np.trace(wcf) / np.sum(wcf)
    else:
        weighted_accuracy = float('nan')  # Handle division by zero

    # Return the metric in a dictionary
    return {'weighted_accuracy': weighted_accuracy}
