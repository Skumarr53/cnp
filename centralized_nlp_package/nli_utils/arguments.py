# nli_finetune/arguments.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data for training and evaluation.

    Attributes:
        task_name (Optional[str]): The name of the task to train on.
        dataset_name (Optional[str]): The name of the dataset to use (via the datasets library).
        dataset_config_name (Optional[str]): The configuration name of the dataset to use.
        max_seq_length (int): The maximum total input sequence length after tokenization.
        overwrite_cache (bool): Overwrite the cached preprocessed datasets or not.
        pad_to_max_length (bool): Whether to pad all samples to 'max_seq_length'.
        max_train_samples (Optional[int]): Truncate the number of training examples for debugging or quicker training.
        max_eval_samples (Optional[int]): Truncate the number of evaluation examples for debugging or quicker evaluation.
        max_predict_samples (Optional[int]): Truncate the number of prediction examples for debugging or quicker prediction.
        train_file (Optional[str]): Path to a CSV or JSON file containing the training data.
        validation_file (Optional[str]): Path to a CSV or JSON file containing the validation data.
        test_file (Optional[str]): Path to a CSV or JSON file containing the test data.

    Usage Example:
        >>> from nli_utils.arguments import DataTrainingArguments
        >>> data_args = DataTrainingArguments(
        ...     task_name="mnli",
        ...     max_seq_length=128,
        ...     train_file="path/to/train.csv",
        ...     validation_file="path/to/validation.csv"
        ... )
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: mnli, cola, etc."},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to 'max_seq_length'. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value for debugging or quicker training."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples to this value for debugging or quicker evaluation."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples to this value for debugging or quicker prediction."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in {
                "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
            }:
                raise ValueError(
                    "Unknown task, please select one from: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli."
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "You must specify either a GLUE task, a training/validation file, or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "'train_file' should be a CSV or a JSON file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "'validation_file' should have the same extension as 'train_file'."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to the model configuration.

    Attributes:
        model_name_or_path (str): Path to pretrained model or model identifier from huggingface.co/models.
        config_name (Optional[str]): Pretrained config name or path if not the same as model_name.
        tokenizer_name (Optional[str]): Pretrained tokenizer name or path if not the same as model_name.
        cache_dir (Optional[str]): Directory to store the pretrained models downloaded from huggingface.co.
        use_fast_tokenizer (bool): Whether to use a fast tokenizer (backed by the tokenizers library) or not.
        model_revision (str): The specific model version to use (can be a branch name, tag name, or commit id).
        token (Optional[str]): The token for HTTP bearer authorization for remote files.
        trust_remote_code (bool): Whether to allow custom models defined on the Hub in their own modeling files.
        ignore_mismatched_sizes (bool): Enable loading a pretrained model whose head dimensions are different.

    Usage Example:
        >>> from nli_utils.arguments import ModelArguments
        >>> model_args = ModelArguments(
        ...     model_name_or_path="bert-base-uncased",
        ...     cache_dir="./cache",
        ... )
    """
    model_name_or_path: str = field(
        default="/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name, or commit id)."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token for HTTP bearer authorization for remote files. If not specified, will use the token "
                "from `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to allow custom models defined on the Hub in their own modeling files. "
                "Set to 'True' only for trusted repositories."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Enable loading a pretrained model whose head dimensions are different."},
    )
