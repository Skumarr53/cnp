import pandas as pd
import os
#from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_and_plot_confusion_matrices(data, label_col, prediction_col, topic_col, plot_func):
    """
    Generate and plot confusion matrices for the overall dataset and individual topics.

    This function computes the confusion matrix for the entire dataset and for each unique topic within the dataset. It utilizes the provided plotting function to visualize and save each confusion matrix. The filenames of the generated plots are returned for further use or logging.

    Args:
        data (pd.DataFrame):
            The DataFrame containing the evaluation results with true labels and predictions.
        label_col (str):
            The column name in `data` that contains the ground truth labels.
        prediction_col (str):
            The column name in `data` that contains the predicted labels.
        topic_col (str):
            The column name in `data` that contains topic identifiers for generating per-topic confusion matrices.
        plot_func (Callable[[Any, str, str], str]):
            A function that takes a confusion matrix, a title, and a filename as input, generates the plot, saves it to the specified filename, and returns the filename.

    Returns:
        List[str]:
            A list of filenames for the generated confusion matrix plots.

    Raises:
        KeyError:
            If any of the specified columns (`label_col`, `prediction_col`, `topic_col`) are not present in the DataFrame.
        ValueError:
            If the DataFrame is empty or does not contain any unique topics.

    Example:
        >>> import pandas as pd
        >>> def custom_plot(matrix, title, filename):
        ...     import matplotlib.pyplot as plt
        ...     import seaborn as sns
        ...     plt.figure(figsize=(6,4))
        ...     sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        ...     plt.title(title)
        ...     plt.xlabel('Predicted')
        ...     plt.ylabel('Actual')
        ...     plt.savefig(filename)
        ...     plt.close()
        ...     return filename
        >>> data = pd.DataFrame({
        ...     'label': ['entailment', 'contradiction', 'neutral', 'entailment'],
        ...     'prediction': ['entailment', 'neutral', 'neutral', 'contradiction'],
        ...     'topic': ['topic1', 'topic1', 'topic2', 'topic2']
        ... })
        >>> filenames = generate_and_plot_confusion_matrices(data, 'label', 'prediction', 'topic', custom_plot)
        >>> print(filenames)
        ['overall_confusion_matrix.png', 'confusion_matrix_topic1.png', 'confusion_matrix_topic2.png']
    """
    # Generate the overall confusion matrix
    overall_conf_matrix = confusion_matrix(data[label_col], data[prediction_col])
    
    # Generate confusion matrices for each individual topic
    plot_filnames = []
    topic_conf_matrices = {}
    for topic in data[topic_col].unique():
        topic_data = data[data[topic_col] == topic]
        topic_conf_matrix = confusion_matrix(topic_data[label_col], topic_data[prediction_col])
        topic_conf_matrices[topic] = topic_conf_matrix
    
    # Display and log the overall confusion matrix
    overall_plot_name = plot_func(overall_conf_matrix, 'Overall Confusion Matrix', 'overall_confusion_matrix.png')
    plot_filnames.append(overall_plot_name)
    # Display and log confusion matrices for each topic
    
    for topic, matrix in topic_conf_matrices.items():
        plot_name = plot_func(matrix, f"Confusion Matrix for Topic: {topic}", f"confusion_matrix_{topic.replace(' ','_')}.png")
        plot_filnames.append(plot_name)
    return plot_filnames

def plot_conf_matrix(matrix, title, filename):
    """
    Plot a confusion matrix using seaborn's heatmap and save it as an artifact.

    This function visualizes the provided confusion matrix with annotations and saves the plot to the specified filename. It ensures the plot is closed after saving to free up memory resources.

    Args:
        matrix (Any):
            The confusion matrix to plot, typically a 2D array or matrix structure.
        title (str):
            The title of the confusion matrix plot.
        filename (str):
            The filename (including path) where the plot image will be saved.

    Returns:
        str:
            The filename of the saved confusion matrix plot.

    Raises:
        ValueError:
            If the provided `matrix` is not in a plottable format.
        IOError:
            If the plot cannot be saved to the specified `filename`.

    Example:
        >>> import numpy as np
        >>> def mock_log_artifact(file_path):
        ...     print(f"Artifact logged: {file_path}")
        >>> sns = __import__('seaborn')
        >>> plt = __import__('matplotlib.pyplot')
        >>> matrix = np.array([[50, 2], [5, 43]])
        >>> filename = plot_conf_matrix(matrix, "Confusion Matrix Example", "confusion_matrix_example.png")
        >>> print(filename)
        confusion_matrix_example.png
        Artifact logged: confusion_matrix_example.png
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()

    return filename