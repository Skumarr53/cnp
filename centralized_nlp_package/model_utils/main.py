from centralized_nlp_package.mlflow_utils import ExperimentManager, ModelSelector, ModelTransition, get_current_date
from loguru import logger

def main():
    # Define experiment parameters
    base_name = "Fine-tune_DeBERTa_v3"
    data_src = "CallTranscript"
    run_date = get_current_date()
    experiment_name = f"{base_name}_{data_src}_{run_date}"
    
    dataset_versions = ["version1", "version2"]  # Customizable
    hyperparameters = [
        {"n_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16, "eval_batch_size": 16},
        {"n_epochs": 10, "learning_rate": 3e-5, "weight_decay": 0.02, "train_batch_size": 32, "eval_batch_size": 32}
    ]
    base_model = "deberta"  # or "finbert"
    model_path = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"
    output_dir = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3"
    train_file = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_train_v3.csv"
    validation_file = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/NLI_role_training_data/Role_FT_test_v3.csv"
    
    # Initialize ExperimentManager
    experiment_manager = ExperimentManager(
        base_name=base_name,
        data_src=data_src,
        dataset_versions=dataset_versions,
        hyperparameters=hyperparameters,
        base_model=base_model,
        model_path=model_path,
        output_dir=output_dir,
        train_file=train_file,
        validation_file=validation_file
    )
    
    # Run experiments
    experiment_manager.run_experiments()
    
    # Initialize ModelSelector
    selector = ModelSelector(experiment_name=experiment_manager.experiment_name, metric="accuracy")
    best_run = selector.get_best_model()
    
    if best_run:
        # Initialize ModelTransition
        model_transition = ModelTransition(model_name="Fine-tuned_Model")
        # Transition the best model to Production (or any other stage)
        model_transition.transition_model(stage="Production", experiment_name=experiment_manager.experiment_name)
    else:
        logger.warning("No best model found to transition.")

if __name__ == "__main__":
    main()