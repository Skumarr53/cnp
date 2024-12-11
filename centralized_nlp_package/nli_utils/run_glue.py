# nli_finetune/run_glue.py

#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import time
from typing import Optional, Dict, Any, Tuple
from loguru import logger
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from centralized_nlp_package.nli_utils import (
    DataTrainingArguments,
    ModelArguments,
    prepare_datasets,
    preprocess_datasets,
    get_compute_metrics,
    setup_logging,
    initialize_trainer,
    train,
    evaluate,
    predict,
)

def run_glue(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> None:
    
    """
    Executes the GLUE task by orchestrating data preparation, model training, evaluation, and prediction.

    Args:
        model_args (ModelArguments): Configuration related to the model.
        data_args (DataTrainingArguments): Configuration related to data processing.
        training_args (TrainingArguments): Configuration related to training.

    Usage Example:
        >>> from centralized_nlp_package.nli_utils import run_glue
        >>> from centralized_nlp_package.nli_utils import DataTrainingArguments
        >>> from centralized_nlp_package.nli_utils import ModelArguments
        >>> from transformers import TrainingArguments
        >>> 
        >>> model_args = ModelArguments(
        ...     model_name_or_path="bert-base-uncased",
        ...     cache_dir="./cache",
        ... )
        >>> data_args = DataTrainingArguments(
        ...     task_name="mnli",
        ...     train_file="path/to/train.csv",
        ...     validation_file="path/to/validation.csv",
        ... )
        >>> training_args = TrainingArguments(
        ...     output_dir="./output",
        ...     do_train=True,
        ...     do_eval=True,
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ...     weight_decay=0.01,
        ...     per_device_train_batch_size=16,
        ...     per_device_eval_batch_size=16,
        ...     report_to="none",
        ... )
        >>> run_glue(model_args, data_args, training_args)
    """

    setup_logging(training_args)

    logger.info("Starting GLUE task...")
    logger.info(f"Training/evaluation parameters: {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use 'overwrite_output_dir' to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected at {last_checkpoint}. Resuming training from checkpoint."
            )

    set_seed(training_args.seed)

    raw_datasets = prepare_datasets(data_args, model_args, training_args)

    if data_args.task_name is not None:
        is_regression = data_args.task_name.lower() == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        learning_rate = training_args.learning_rate,
        weight_decay = training_args.weight_decay,
        num_train_epochs = training_args.num_train_epochs
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,

    )

    ## add additional parameters to config 
    config.learning_rate = training_args.learning_rate
    config.weight_decay = training_args.weight_decay
    config.num_train_epochs = training_args.num_train_epochs
    config.train_batch_size = training_args.per_device_train_batch_size
    config.eval_batch_size = training_args.per_device_eval_batch_size
    config.model_family = model_args.model_name_or_path.split('/')[-1]

    config.save_pretrained(training_args.output_dir)


    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


    raw_datasets = preprocess_datasets(model, raw_datasets, data_args, training_args, tokenizer, num_labels, is_regression, config, label_list)


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    ## Add inference time
    if training_args.do_eval:
        
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    if training_args.do_predict and data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    if data_args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    elif training_args.fp16:
        data_collator = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    compute_metrics = get_compute_metrics(is_regression, data_args.task_name)

    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trained_model = None
    eval_metrics = {}

    if training_args.do_train:
        train_metrics = train(trainer, data_args, model_args, training_args)
        print(f"***** Train metrics *****")
        print(train_metrics)
        eval_metrics.update(train_metrics)

    if training_args.do_eval:
        start_time = time.time()
        evaluation = evaluate(trainer, data_args, model_args, task_name=data_args.task_name)
        eval_execution_time = time.time() - start_time
        print(f"***** evaluate metrics *****")
        eval_metrics['evalaution_time'] = eval_execution_time
        eval_metrics.update(evaluation)

    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

    trained_model = trainer.model
    print(eval_metrics)
    return trained_model, tokenizer, eval_metrics

def run_finetune(
    base_model_path: str,
    train_file: str,
    output_dir: str,
    validation_file: Optional[str] = None,
    param_dict: Optional[Dict[str, Any]] = None
) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
    """
    Fine-tunes a pre-trained model on a given dataset.

    Args:
        base_model_path (str): Path to the base pre-trained model.
        train_file (str): Path to the training data file.
        output_dir (str): Directory to save the fine-tuned model and outputs.
        validation_file (Optional[str], optional): Path to the validation data file. Defaults to None.
        param_dict (Optional[Dict[str, Any]], optional): Dictionary of custom hyperparameters. Defaults to None.

    Returns:
        Tuple[AutoModelForSequenceClassification, Dict[str, float]]: The trained model and evaluation metrics.
    """
    if param_dict is None:
        param_dict = {}

    logger.info("Starting fine-tuning process for the model.")

    try:
        # Prepare ModelArguments
        model_args = ModelArguments(
            model_name_or_path=base_model_path,
            cache_dir=param_dict.get("cache_dir")
        )
        logger.debug(f"ModelArguments: {model_args}")

        # Prepare DataTrainingArguments
        data_args = DataTrainingArguments(
            task_name=param_dict.get("task_name", None),
            train_file=train_file,
            validation_file=validation_file,
            max_seq_length=param_dict.get("max_seq_length", 128),
            pad_to_max_length=param_dict.get("pad_to_max_length", True),
            overwrite_cache=param_dict.get("overwrite_cache", False),
            max_train_samples=param_dict.get("max_train_samples"),
            max_eval_samples=param_dict.get("max_eval_samples"),
            max_predict_samples=param_dict.get("max_predict_samples")
        )
        logger.debug(f"DataTrainingArguments: {data_args}")

        # Prepare TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            num_train_epochs=param_dict.get("n_epochs", 3),
            learning_rate=param_dict.get("learning_rate", 2e-5),
            weight_decay=param_dict.get("weight_decay", 0.01),
            per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            per_device_eval_batch_size=param_dict.get("eval_batch_size", 16),
            fp16=param_dict.get("fp16", True),
            report_to=param_dict.get("report_to", "none"),
            overwrite_output_dir=param_dict.get("overwrite_output_dir", True),
            push_to_hub=param_dict.get("push_to_hub", False),
            seed=param_dict.get("seed", 42),
            logging_dir=param_dict.get("logging_dir", os.path.join(output_dir, "logs")),
            logging_steps=param_dict.get("logging_steps", 500),
            evaluation_strategy=param_dict.get("evaluation_strategy", "steps"),
            save_strategy=param_dict.get("save_strategy", "steps"),
            save_steps=param_dict.get("save_steps", 1000),
            load_best_model_at_end=param_dict.get("load_best_model_at_end", True),
            metric_for_best_model=param_dict.get("metric_for_best_model", "accuracy")
        )
        logger.debug(f"TrainingArguments: {training_args}")

        # Call run_glue (ensure that run_glue is correctly implemented)
        finetuned_model, tokenizer, eval_metrics = run_glue(model_args, data_args, training_args)
        
        logger.info("Fine-tuning completed successfully.")
        logger.info(f"Evaluation Metrics: {eval_metrics}")

        return finetuned_model,tokenizer, eval_metrics

    except Exception as e:
        logger.error(f"An error occurred during fine-tuning: {e}", exc_info=True)
        raise

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_glue(model_args, data_args, training_args)

if __name__ == "__main__":
    main()
