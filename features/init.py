"""
Package pour l'extraction de caractéristiques vocales

Modules:
    acoustic   : Extraction des caractéristiques acoustiques
    temporal   : Extraction des caractéristiques temporelles
    linguistic : Extraction des caractéristiques linguistiques
"""

from .acoustic import extract_acoustic_features
from .temporal import extract_temporal_features
from .linguistic import extract_linguistic_features

__all__ = [
    'extract_acoustic_features',
    'extract_temporal_features',
    'extract_linguistic_features'
]