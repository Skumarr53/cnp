# nli_finetune/data.py

import logging
from typing import Dict, Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import PreTrainedTokenizer, TrainingArguments, PretrainedConfig
from centralized_nlp_package.nli_utils import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def prepare_datasets(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,  # Forward reference
    training_args: TrainingArguments,
) -> datasets.DatasetDict:
    """
    Load datasets based on the provided arguments.

    Args:
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.

    Returns:
        datasets.DatasetDict: Loaded datasets.
    
    Usage Example:
        >>> from nli_finetune.arguments import DataTrainingArguments, ModelArguments
        >>> from nli_finetune.data import prepare_datasets
        >>> data_args = DataTrainingArguments(task_name="mnli")
        >>> model_args = ModelArguments(model_name_or_path="bert-base-uncased")
        >>> datasets = prepare_datasets(data_args, model_args)
    """
    if data_args.task_name is not None:
        # Load a dataset from the GLUE benchmark
        logger.info(f"Loading GLUE task '{data_args.task_name}'")
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Load a dataset from the Hugging Face Hub
        logger.info(f"Loading dataset '{data_args.dataset_name}' with config '{data_args.dataset_config_name}'")
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Load a dataset from local files
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    
    logger.info("Datasets loaded successfully.")
    return raw_datasets

def preprocess_datasets(
    model: transformers.PreTrainedModel,
    raw_datasets: datasets.DatasetDict,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    num_labels: int,
    is_regression: bool,
    config,
    label_list
) -> datasets.DatasetDict:
    """
    Tokenize and preprocess the datasets.

    Args:
        raw_datasets (datasets.DatasetDict): The raw datasets.
        data_args (DataTrainingArguments): Data-related arguments.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        datasets.DatasetDict: Tokenized and preprocessed datasets.
    
    Usage Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> tokenized_datasets = preprocess_datasets(raw_datasets, data_args, tokenizer)
    """
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Infer sentence keys from the dataset
        print(f"train columns: {raw_datasets['train'].column_names}")
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if data_args.pad_to_max_length else False
    
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    logger.info("Datasets tokenized successfully.")
    return tokenized_datasets
