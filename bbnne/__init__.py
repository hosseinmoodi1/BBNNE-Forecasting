"""
BBNNE: Bat-Based Boosted Neural Network Ensemble with Relevant Lag
Space Generation (RLSG).
"""
from .model import BBNNE
from .ablation import BBNNEAblation
from .rlsg import rlsg, compute_relevance_scores
from .bat_algorithm import BatAlgorithm
from .neural_network import MLPExpert
from .features import build_features
from .metrics import (
    evaluate,
    mape,
    mae,
    rmse,
    directional_accuracy,
    theils_u,
    r_squared,
)

__version__ = "1.0.0"
__all__ = [
    "BBNNE",
    "BBNNEAblation",
    "rlsg",
    "compute_relevance_scores",
    "BatAlgorithm",
    "MLPExpert",
    "build_features",
    "evaluate",
    "mape",
    "mae",
    "rmse",
    "directional_accuracy",
    "theils_u",
    "r_squared",
]