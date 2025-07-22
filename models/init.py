"""
Package pour la modélisation et l'interprétation

Modules:
    train_model    : Entraînement des modèles individuels
    explain_model  : Explication des prédictions avec SHAP
    stacking_model : Modèle de stacking ensembliste
"""

from .train_model import main as train_classifier
from .explain_model import main as explain_predictions
from .stacking_model import main as train_stacking_model

__all__ = [
    'train_classifier',
    'explain_predictions',
    'train_stacking_model'
]