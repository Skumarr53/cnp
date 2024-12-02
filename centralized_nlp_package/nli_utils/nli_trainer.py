# nli_finetune/trainer.py

import logging
import os
from typing import Optional, Dict

import numpy as np
import transformers 
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, PreTrainedTokenizer
from transformers import default_data_collator, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import set_seed

from .metrics import get_compute_metrics
from .data import prepare_datasets, preprocess_datasets
from .arguments import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)

def setup_logging(training_args: TrainingArguments):
    """
    Configure logging for the training process.

    Args:
        training_args (TrainingArguments): Training-related arguments.

    Usage Example:
        >>> setup_logging(training_args)
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    transformers.utils.logging.set_verbosity_info() if training_args.should_log else transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def initialize_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizer,
    data_collator,
    compute_metrics,
) -> Trainer:
    """
    Initialize the Hugging Face Trainer.

    Args:
        model (AutoModelForSequenceClassification): The model to train.
        training_args (TrainingArguments): Training-related arguments.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        data_collator: Data collator for the trainer.
        compute_metrics: Function to compute metrics.

    Returns:
        Trainer: An initialized Trainer instance.

    Usage Example:
        >>> trainer = initialize_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics)
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def train(
    trainer: Trainer,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments
) -> None:
    """
    Train the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
    
    Returns:
        Dict[str, float]: Training metrics.

    Usage Example:
        >>> train(trainer, data_args, model_args)
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    print(f"last checkpoint: {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too

    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset) if data_args.max_train_samples is None else min(data_args.max_train_samples, len(trainer.train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return metrics

def evaluate(
    trainer: Trainer, 
    data_args: DataTrainingArguments, 
    model_args: ModelArguments, 
    task_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
        task_name (Optional[str]): The name of the task for evaluation.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(trainer.eval_dataset) if data_args.max_eval_samples is None else min(data_args.max_eval_samples, len(trainer.eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics

def predict(trainer: Trainer, training_args, task_name: Optional[str] = None) -> None:
    """
    Run predictions using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
        task_name (Optional[str]): The name of the task for prediction.

    Usage Example:
        >>> predict(trainer, data_args, model_args, task_name="mnli")
    """
    logger.info("*** Predict ***")
    predictions = trainer.predict(trainer.predict_dataset).predictions
    is_regression = task_name == "stsb" if task_name else False
    predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task_name}.txt" if task_name else "predict_results.txt")
    if is_main_process(training_args.local_rank):
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {task_name if task_name else ''} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:.3f}\n")
                else:
                    label = trainer.model.config.id2label[item] if hasattr(trainer.model.config, "id2label") else item
                    writer.write(f"{index}\t{label}\n")
