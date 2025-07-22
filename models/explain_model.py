import argparse
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main(classification_id, data_type, model_type='best'):
    # Chargement des données
    data_path = f"data/classifications/classification_{classification_id}_{data_type}.csv"
    data = pd.read_csv(data_path)
    X = data.drop('deficit_class', axis=1)
    y = data['deficit_class']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Chargement du modèle
    model_dir = f"models/classif_{classification_id}"
    
    if model_type == 'best':
        model_path = os.path.join(model_dir, f"best_model_{data_type}.pkl")
    else:
        model_path = os.path.join(model_dir, f"{model_type}_{data_type}.pkl")
    
    model = joblib.load(model_path)
    
    # Création de l'explainer SHAP
    sample = X_train.sample(min(100, len(X_train)), random_state=42)
    
    # Préparation des données pour SHAP
    if 'scaler' in model.named_steps:
        scaled_sample = model.named_steps['scaler'].transform(sample)
    else:
        scaled_sample = sample
    
    # Sélection de l'explainer selon le type de modèle
    model_name = model_type if model_type != 'best' else 'best_model'
    
    if 'RandomForest' in model_path or 'XGBoost' in model_path:
        explainer = shap.TreeExplainer(model.named_steps['model'])
        shap_values = explainer.shap_values(scaled_sample)
    elif 'LogReg' in model_path:
        explainer = shap.LinearExplainer(
            model.named_steps['model'], 
            scaled_sample,
            feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(scaled_sample)
    else:
        explainer = shap.KernelExplainer(
            model.named_steps['model'].predict_proba, 
            shap.kmeans(scaled_sample, 10)
        )
        shap_values = explainer.shap_values(scaled_sample)[1]  # Classe positive
    
    # Visualisation
    output_dir = f"results/classif_{classification_id}/shap/{data_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, scaled_sample, feature_names=X.columns, 
                      plot_type="bar", max_display=15, show=False)
    plt.title(f"Top Features - {data_type} ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_top_features.png"), dpi=300)
    plt.close()
    
    # Summary plot (dot)
    plt.figure()
    shap.summary_plot(shap_values, scaled_sample, feature_names=X.columns, 
                      max_display=15, show=False)
    plt.title(f"Feature Impact - {data_type} ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_feature_impact.png"), dpi=300)
    plt.close()
    
    # Dependence plots pour les top 5 features
    top_features = np.abs(shap_values).mean(0).argsort()[-5:][::-1]
    for i, idx in enumerate(top_features):
        plt.figure()
        shap.dependence_plot(
            idx, shap_values, scaled_sample, 
            feature_names=X.columns,
            interaction_index=None,
            show=False
        )
        plt.title(f"Dependence Plot - {X.columns[idx]} ({data_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_dependence_{X.columns[idx]}.png"), dpi=300)
        plt.close()
    
    # Waterfall plot pour un échantillon représentatif
    sample_idx = np.abs(shap_values.sum(1) - np.mean(shap_values.sum(1))).argmin()
    plt.figure()
    shap.waterfall_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[sample_idx], 
        features=scaled_sample[sample_idx],
        feature_names=X.columns,
        max_display=10,
        show=False
    )
    plt.title(f"Waterfall Plot - Sample {sample_idx} ({data_type})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_waterfall_sample.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for models')
    parser.add_argument("--classification", type=int, required=True, 
                        choices=range(1, 11), help='Clinical deficit ID (1-10)')
    parser.add_argument("--type", choices=['NEG', 'POS'], required=True,
                        help='Data type: NEG (negative) or POS (positive)')
    parser.add_argument("--model", type=str, default='best',
                        choices=['best', 'RandomForest', 'XGBoost', 'LogReg', 'SVM'],
                        help='Model type to explain')
    args = parser.parse_args()
    
    main(args.classification, args.type, args.model)