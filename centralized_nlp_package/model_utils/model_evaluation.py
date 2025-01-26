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
    Performs K-Fold cross-validation training for NLI tasks using specified model and hyperparameters.

    Parameters:
    - data_path (str): Path to the CSV data file.
    - base_exp_name (str): Base name for the experiment.
    - data_src (str): Data source identifier.
    - model_version (str): Model version to use.
    - hyperparameters (dict): Hyperparameters for training.
    - user_id (str): User ID for the experiment.
    - n_splits (int): Number of splits for K-Fold cross-validation.
    - random_state (int): Random state for reproducibility.
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
    Generates and plots confusion matrices for overall data and individual topics.

    Parameters:
    - data: DataFrame containing the data.
    - label_col: Column name for ground truth labels.
    - prediction_col: Column name for predicted labels.
    - topic_col: Column name for topics to generate individual confusion matrices.
    - plot_func: Function to plot the confusion matrix.

    Returns:
    - overall_conf_matrix: Confusion matrix for the entire dataset.
    - topic_conf_matrices: Dictionary of confusion matrices for each topic.
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
    Plots a confusion matrix using seaborn heatmap and logs it as an artifact in MLflow.

    Parameters:
    - matrix: Confusion matrix to plot.
    - title: Title for the plot.
    - filename: Filename to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()

    return filename