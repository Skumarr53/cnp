import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger


def initialize_nli_infer_pipeline(model_path: str, enable_quantization=False):
    """
    Initialize a Natural Language Inference (NLI) inference pipeline with optional model quantization.
    
    This function sets up a Hugging Face Transformers pipeline for zero-shot classification using the specified model. It supports dynamic quantization to optimize the model for faster inference and reduced memory usage. The pipeline is configured to run on a GPU if available; otherwise, it defaults to the CPU.
    
    Args:
        model_path (str):
            The file path or identifier of the pre-trained model to be used for the NLI pipeline. This can be a local directory containing the model files or a model identifier from the Hugging Face Model Hub.
        
        enable_quantization (bool, optional):
            Flag indicating whether to apply dynamic quantization to the model's linear layers. Quantization can reduce the model's memory footprint and potentially increase inference speed, especially on CPU. Defaults to `False`.
    
    Returns:
        transformers.pipelines.Pipeline:
            A Hugging Face Transformers pipeline configured for zero-shot classification, ready to perform NLI tasks.
    
    Raises:
        FileNotFoundError:
            If the specified `model_path` does not exist or is inaccessible.
        
        ValueError:
            If the model at `model_path` is incompatible with the zero-shot classification task.
        
        Exception:
            For any other errors that occur during model loading, quantization, or pipeline initialization.
    
    Example:
        ```python
        from nli_finetune.metrics import initialize_nli_infer_pipeline
        
        # Initialize the NLI pipeline without quantization
        nli_pipeline = initialize_nli_infer_pipeline(
            model_path="facebook/bart-large-mnli",
            enable_quantization=False
        )
        
        # Perform a zero-shot classification inference
        result = nli_pipeline(
            sequences="The movie was fantastic and I loved it.",
            candidate_labels=["positive", "negative", "neutral"],
            hypothesis_template="The sentiment of the review is {}."
        )
        
        print(result)
        # Output:
        # {
        #     'sequence': 'The movie was fantastic and I loved it.',
        #     'labels': ['positive', 'neutral', 'negative'],
        #     'scores': [0.95, 0.04, 0.01]
        # }
        ```
    
    Notes:
        - **Quantization**: Enabling quantization (`enable_quantization=True`) applies dynamic quantization to the model's linear layers, converting them to a lower precision (e.g., `torch.qint8`). This can lead to reduced model size and faster inference times, particularly beneficial when deploying models on CPU-only environments. However, quantization may slightly degrade model accuracy.
        
        - **Device Configuration**: The pipeline automatically utilizes a GPU if one is available (`device=0`). If no GPU is detected, it defaults to the CPU (`device=-1`). Ensure that the appropriate hardware is available for optimal performance.
        
        - **Model Compatibility**: The specified `model_path` must support the zero-shot classification task. Models fine-tuned for tasks like sentiment analysis or entailment are suitable for NLI inference pipelines.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        if enable_quantization:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.debug("Model quantization enabled.")
        else:
            logger.debug("Model quantization disabled.")
        
        device = 0 if torch.cuda.is_available() else -1
        nli_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        logger.debug(f"NLI pipeline initialized on device: {'GPU' if device == 0 else 'CPU'}")
        return nli_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize NLI pipeline: {e}")
        raise e

