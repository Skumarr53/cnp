# nli_finetuning/__init__.py

from .arguments import DataTrainingArguments, ModelArguments
from .data import prepare_datasets, preprocess_datasets
from .metrics import get_compute_metrics, evaluate_nli_models, compute_metrics, get_nli_model_metrics
from .nli_trainer import (
    setup_logging,
    initialize_trainer,
    train,
    evaluate,
    predict
)
from .run_glue import run_glue, run_finetune
from .nli_inference import initialize_nli_infer_pipeline


__all__ = [
    "ModelArguments",
    "DataTrainingArguments",
    "prepare_datasets",
    "preprocess_datasets",
    "get_compute_metrics",
    "setup_logging",
    "initialize_trainer",
    "train",
    "evaluate",
    "predict",
    "run_glue",
    "initialize_nli_infer_pipeline",
    "evaluate_nli_models",
    "run_finetune",
    "get_nli_model_metrics",
    "compute_metrics"
]


# nli_finetune/__init__.py

"""
nli_finetune
============

A package for fine-tuning and evaluating Natural Language Inference (NLI) models using the GLUE benchmark.

Modules:
- arguments: Defines the data and model argument classes.
- data: Functions for loading and preprocessing datasets.
- metrics: Functions for computing evaluation metrics.
- trainer: Functions to set up and run the training and evaluation processes.
- utils: Utility functions.
- run_glue: Main script to execute the training and evaluation.
"""

__version__ = "1.0.0"
