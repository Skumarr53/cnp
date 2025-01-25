import pandas as pd
import os
from loguru import logger
from sklearn.model_selection import KFold
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
