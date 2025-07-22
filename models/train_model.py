import argparse
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, cohen_kappa_score, log_loss, 
                             roc_curve, precision_recall_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration reproductible
np.random.seed(42)

# Hyperparamètres optimisés pour chaque modèle
MODEL_PARAMS = {
    'RandomForest': {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [None, 10, 20, 30, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False],
        'model__class_weight': ['balanced', None]
    },
    'XGBoost': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 6, 9],
        'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
        'model__subsample': [0.5, 0.8, 1.0],
        'model__colsample_bytree': [0.5, 0.8, 1.0],
        'model__gamma': [0, 0.1, 0.2, 0.5],
        'model__scale_pos_weight': [1, 5, 10]
    },
    'LogReg': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga'],
        'model__class_weight': ['balanced', None]
    },
    'SVM': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
        'model__class_weight': ['balanced', None],
        'model__probability': [True]
    }
}

def save_evaluation_metrics(model, X_test, y_test, output_dir, model_name):
    """Enregistre toutes les métriques et graphiques d'évaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_proba),
        'balanced_accuracy': accuracy_score(y_test, y_pred),
        'precision': f1_score(y_test, y_pred, average='binary', pos_label=1),
        'recall': f1_score(y_test, y_pred, average='binary', pos_label=1)
    }
    
    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics['roc_auc'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()
    
    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % auc(recall, precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'), dpi=300)
    plt.close()
    
    # Matrice de confusion visualisée
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Norm', 'Deficit'],
                yticklabels=['Norm', 'Deficit'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Feature importance pour les modèles basés sur les arbres
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(min(20, len(importances))), importances[indices][:20], align="center")
        plt.xticks(range(min(20, len(importances))), X_test.columns[indices][:20], rotation=90)
        plt.xlim([-1, min(20, len(importances))])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importances.png'), dpi=300)
        plt.close()
    
    # Sauvegarde des résultats
    with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    pd.DataFrame(report).to_csv(os.path.join(output_dir, f'{model_name}_classification_report.csv'))
    pd.DataFrame(cm, 
                columns=['Predicted Norm', 'Predicted Deficit'],
                index=['True Norm', 'True Deficit']
               ).to_csv(os.path.join(output_dir, f'{model_name}_confusion_matrix.csv'))
    
    return metrics

def main(classification_id, data_type):
    # Chargement des données
    data_path = f"data/classifications/classification_{classification_id}_{data_type}.csv"
    data = pd.read_csv(data_path)
    X = data.drop('deficit_class', axis=1)
    y = data['deficit_class'].map({'deficit': 1, 'norm': 0})  # Conversion binaire
    
    # Split stratifié 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Configuration validation croisée 5x5
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    # Modèles à tester
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LogReg": LogisticRegression(max_iter=10000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Random": DummyClassifier(strategy="uniform", random_state=42)
    }
    
    results = {}
    model_dir = f"models/classif_{classification_id}"
    result_dir = f"results/classif_{classification_id}/{data_type}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    best_auc = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Processing {name} for classification {classification_id} {data_type}")
        print(f"{'='*50}")
        
        # Pipeline avec préprocessing et modèle
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        if name == "Random":
            # Pas d'optimisation pour le modèle aléatoire
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            cv_scores = {'test_score': [roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])] * 25}
        else:
            # Recherche aléatoire d'hyperparamètres
            search = RandomizedSearchCV(
                pipe,
                MODEL_PARAMS[name],
                n_iter=50 if name in ['RandomForest', 'XGBoost'] else 30,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=3,
                random_state=42
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            cv_scores = search.cv_results_
            print(f"Best params for {name}: {search.best_params_}")
        
        # Évaluation complète
        test_metrics = save_evaluation_metrics(best_pipe, X_test, y_test, result_dir, name)
        results[name] = test_metrics
        
        # Sauvegarde du modèle
        joblib.dump(best_pipe, os.path.join(model_dir, f"{name}_{data_type}.pkl"))
        
        # Sauvegarde des scores de validation croisée
        cv_df = pd.DataFrame({
            'fold': range(len(cv_scores['mean_test_score'])),
            'auc_score': cv_scores['mean_test_score']
        })
        cv_df.to_csv(os.path.join(result_dir, f"{name}_cv_scores.csv"), index=False)
        
        # Mise à jour du meilleur modèle
        if name != "Random" and test_metrics['roc_auc'] > best_auc:
            best_auc = test_metrics['roc_auc']
            best_model_name = name
            joblib.dump(best_pipe, os.path.join(model_dir, f"best_model_{data_type}.pkl"))
    
    # Sauvegarde des résultats finaux
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(result_dir, "final_metrics.csv"))
    
    print(f"\n{'='*50}")
    print(f"Best model for {data_type}: {best_model_name} with AUC: {best_auc:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train base models for vocal biomarker classification')
    parser.add_argument("--classification", type=int, required=True, 
                        choices=range(1, 11), help='Clinical deficit ID (1-10)')
    parser.add_argument("--type", choices=['NEG', 'POS'], required=True,
                        help='Data type: NEG (negative) or POS (positive)')
    args = parser.parse_args()
    
    main(args.classification, args.type)