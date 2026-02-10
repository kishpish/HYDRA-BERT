"""HYDRA-BERT model components."""

from .hydra_bert import HydraBERT, HydraBERTConfig
from .polybert_lora import PolyBERTWithLoRA, LoRALayer, LoRALinear
from .fusion_transformer import (
    CrossModalFusionTransformer,
    FullFusionModule,
    FormulationEncoder,
    PatientEncoder,
)
from .prediction_heads import (
    GCSRegressionHead,
    OutcomeClassificationHead,
    SafetyHead,
    MultiTaskHead,
)

__all__ = [
    # Main model
    "HydraBERT",
    "HydraBERTConfig",
    # LoRA components
    "PolyBERTWithLoRA",
    "LoRALayer",
    "LoRALinear",
    # Fusion components
    "CrossModalFusionTransformer",
    "FullFusionModule",
    "FormulationEncoder",
    "PatientEncoder",
    # Prediction heads
    "GCSRegressionHead",
    "OutcomeClassificationHead",
    "SafetyHead",
    "MultiTaskHead",
]
