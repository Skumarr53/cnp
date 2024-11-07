# nli_finetune/run_glue.py

#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from typing import Optional
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

    raw_datasets = prepare_datasets(data_args, model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenized_datasets = preprocess_datasets(raw_datasets, data_args, tokenizer)

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

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    if training_args.do_train and data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = (
        tokenized_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if training_args.do_eval
        else None
    )
    if training_args.do_eval and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    predict_dataset = (
        tokenized_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if training_args.do_predict
        else None
    )
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train(trainer, data_args, model_args)

    if training_args.do_eval:
        evaluate(trainer, data_args, model_args, task_name=data_args.task_name)

    if training_args.do_predict:
        predict(trainer, data_args, model_args, task_name=data_args.task_name)

    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_glue(model_args, data_args, training_args)

if __name__ == "__main__":
    main()
