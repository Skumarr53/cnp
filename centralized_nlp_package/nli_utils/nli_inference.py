import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger


def initialize_nli_infer_pipeline(model_path: str, enable_quantization=False):
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

