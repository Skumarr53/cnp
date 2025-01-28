import pandas as pd
import os
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from .experiment_manager import ExperimentManager



def perform_kfold_training(data_path, base_exp_name, data_src, model_version, hyperparameters, user_id, n_splits=5, random_state=42):
    """
    Perform K-Fold cross-validation training for Natural Language Inference (NLI) tasks using the specified model and hyperparameters.

    This function executes K-Fold cross-validation by splitting the dataset into training and testing subsets for each fold. For each split, it initializes an `ExperimentManager` instance to manage the experiment run, conducts training, evaluates the model, and logs the results using MLflow.

    Args:
        data_path (str):
            Path to the CSV data file containing the dataset.
        base_exp_name (str):
            Base name for the experiment, used to construct the MLflow experiment name.
        data_src (str):
            Identifier for the data source.
        model_version (str):
            Version identifier of the base model to be fine-tuned.
        hyperparameters (Dict[str, Any]):
            Dictionary containing hyperparameters for training, such as learning rate, number of epochs, weight decay, and batch size.
        user_id (str):
            User identifier, typically an email address, used in constructing the experiment name.
        n_splits (int, optional):
            Number of folds for K-Fold cross-validation. Defaults to 5.
        random_state (int, optional):
            Seed for random number generator to ensure reproducibility. Defaults to 42.

    Returns:
        None

    Raises:
        FileNotFoundError:
            If the data file specified by `data_path` does not exist.
        pd.errors.EmptyDataError:
            If the data file is empty.
        Exception:
            If any error occurs during data loading, splitting, or experiment execution.

    Example:
        >>> perform_kfold_training(
        ...     data_path="data/nli_dataset.csv",
        ...     base_exp_name="NLI_Experiment",
        ...     data_src="source_A",
        ...     model_version="bert-base-uncased",
        ...     hyperparameters={
        ...         "n_epochs": 3,
        ...         "learning_rate": 2e-5,
        ...         "weight_decay": 0.01,
        ...         "train_batch_size": 16
        ...     },
        ...     user_id="user@example.com",
        ...     n_splits=5,
        ...     random_state=42
        ... )
        INFO - Initialized KFold with 5 splits.
        INFO - Fold 0: Training and test data prepared.
        INFO - Experiment set to /Users/user@example.com/NLI_Experiment_source_A_20250127
        INFO - Starting finetuning run: bert-base-uncased_dataset_v1_param_set1
        ...
        INFO - Fold 0: Experiment completed.
        ...
    """
    # Load data
    data = pd.read_csv(data_path)
    grouped_data = data.groupby('sentence1')
    pairs = [group for _, group in grouped_data]

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info(f"Initialized KFold with {n_splits} splits.")


    # Get the directory of the input data
    input_dir = os.path.dirname(data_path)

    # Iterate over each fold
    for fold, (train_index, test_index) in enumerate(kf.split(pairs)):
        train_pairs = [pairs[i] for i in train_index]
        test_pairs = [pairs[i] for i in test_index]
        print(f"Processing fold {fold}")

        # Concatenate the pairs back into DataFrames
        train_data = pd.concat(train_pairs).reset_index(drop=True)
        test_data = pd.concat(test_pairs).reset_index(drop=True)
        logger.info(f"Fold {fold}: Training and test data prepared.")


        # Save the split data to temporary files in the same directory as the input data
        train_fold = f"train_fold_{fold}.csv"
        train_file_path = os.path.join(input_dir, train_fold)
        test_file_path = os.path.join(input_dir, f"test_fold_{fold}.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)


        # Initialize the ExperimentManager for this fold
        experiment_manager = ExperimentManager(
            base_name=base_exp_name,
            data_src=data_src,
            dataset_versions=[train_fold],
            hyperparameters=[hyperparameters],
            base_model_versions=[model_version],
            train_file=train_file_path,
            validation_file=test_file_path,
            evalute_pretrained_model=False,
            eval_entailment_thresold=0.8,
            user_id=user_id
        )


        # Run the experiment for this fold
        experiment_manager.run_experiments()
        logger.info(f"Fold {fold}: Experiment completed.")


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
    topic_conf_matrices = {}
    for topic in data[topic_col].unique():
        topic_data = data[data[topic_col] == topic]
        topic_conf_matrix = confusion_matrix(topic_data[label_col], topic_data[prediction_col])
        topic_conf_matrices[topic] = topic_conf_matrix
    
    # Display and log the overall confusion matrix
    plot_func(overall_conf_matrix, 'Overall Confusion Matrix', 'overall_confusion_matrix.png')
    
    # Display and log confusion matrices for each topic
    plot_filnames = []
    for topic, matrix in topic_conf_matrices.items():
        plot_name = plot_func(matrix, f'Confusion Matrix for Topic: {topic}', f'confusion_matrix_{topic}.png')
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