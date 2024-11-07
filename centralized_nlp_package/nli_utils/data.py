# nli_finetune/data.py

import logging
from typing import Dict, Optional

import datasets
import transformers
from datasets import load_dataset
from transformers import PreTrainedTokenizer
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
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Load a dataset from the Hugging Face Hub
        logger.info(f"Loading dataset '{data_args.dataset_name}' with config '{data_args.dataset_config_name}'")
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=model_args.token,
        )
    else:
        # Load a dataset from local files
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file

        extension = data_args.train_file.split(".")[-1]
        logger.info(f"Loading local dataset with extension '{extension}'")
        if extension == "csv":
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir, use_auth_token=model_args.token)
        elif extension == "json":
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir, use_auth_token=model_args.token)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    logger.info("Datasets loaded successfully.")
    return raw_datasets

def preprocess_datasets(
    raw_datasets: datasets.DatasetDict,
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
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
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    logger.info(f"Tokenizing datasets with padding='{padding}' and max_seq_length={max_seq_length}")

    def tokenize_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Remove columns that are not needed after tokenization
    if "train" in raw_datasets:
        columns_to_remove = [col for col in raw_datasets["train"].column_names if col not in {sentence1_key, sentence2_key, "label"}]
    else:
        columns_to_remove = [col for col in raw_datasets.column_names if col not in {sentence1_key, sentence2_key, "label"}]

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing the datasets",
    )

    logger.info("Datasets tokenized successfully.")
    return tokenized_datasets
