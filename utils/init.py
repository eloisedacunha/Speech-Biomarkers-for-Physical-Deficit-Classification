"""
Package utilitaire pour la configuration et le reporting

Modules:
    config            : Configuration centrale du projet
    report_generator  : Génération de rapports d'évaluation
"""

from .config import DEFICIT_MAP, CLINICAL_DEFICITS, ACOUSTIC_FEATURES, LINGUISTIC_FEATURES, TEMPORAL_FEATURES, DATA_PATHS
from .report_generator import generate_model_report, generate_shap_report

__all__ = [
    'DEFICIT_MAP',
    'CLINICAL_DEFICITS',
    'ACOUSTIC_FEATURES',
    'LINGUISTIC_FEATURES',
    'TEMPORAL_FEATURES',
    'DATA_PATHS',
    'generate_model_report',
    'generate_shap_report'
]