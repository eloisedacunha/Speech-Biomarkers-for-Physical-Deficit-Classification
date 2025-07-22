import argparse
import joblib
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                            classification_report, cohen_kappa_score, log_loss,
                            roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils.config import DEFICIT_MAP

# Configuration reproductible
np.random.seed(42)

# Hyperparamètres pour les méta-modèles
META_PARAMS = {
    'RandomForest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.01, 0.1]
    },
    'LogReg': {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2']
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }
}

META_MODELS = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LogReg': LogisticRegression(max_iter=10000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

def save_evaluation_metrics(model, X_test, y_test, output_dir, model_name):
    """Enregistre les métriques et graphiques d'évaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_proba),
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Stacking')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()
    
    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Stacking')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'{model_name}_pr_curve.png'), dpi=300)
    plt.close()
    
    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Norm', 'Deficit'],
                yticklabels=['Norm', 'Deficit'])
    plt.title('Confusion Matrix - Stacking')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Sauvegarde des résultats
    with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    pd.DataFrame(report).to_csv(os.path.join(output_dir, f'{model_name}_classification_report.csv'))
    pd.DataFrame(cm).to_csv(os.path.join(output_dir, f'{model_name}_confusion_matrix.csv'))
    
    return metrics

def main(classification_id):
    deficit_name = DEFICIT_MAP[classification_id]
    print(f"Training stacking model for {deficit_name} (classification {classification_id})")
    
    # Chargement des données
    neg_data = pd.read_csv(f"data/classifications/classification_{classification_id}_NEG.csv")
    pos_data = pd.read_csv(f"data/classifications/classification_{classification_id}_POS.csv")
    
    # Vérification de l'alignement des cibles
    y = neg_data['deficit_class'].map({'deficit': 1, 'norm': 0})
    assert y.equals(pos_data['deficit_class'].map({'deficit': 1, 'norm': 0})), "Target mismatch"
    
    # Chargement des meilleurs modèles de base
    model_dir = f"models/classif_{classification_id}"
    best_neg = joblib.load(os.path.join(model_dir, "best_model_NEG.pkl"))
    best_pos = joblib.load(os.path.join(model_dir, "best_model_POS.pkl"))
    
    # Création des features de stacking
    X_neg = neg_data.drop('deficit_class', axis=1)
    X_pos = pos_data.drop('deficit_class', axis=1)
    
    proba_neg = best_neg.predict_proba(X_neg)[:, 1]
    proba_pos = best_pos.predict_proba(X_pos)[:, 1]
    
    # Features supplémentaires: prédictions brutes et métriques
    pred_neg = best_neg.predict(X_neg)
    pred_pos = best_pos.predict(X_pos)
    
    X_stack = pd.DataFrame({
        'NEG_proba': proba_neg,
        'POS_proba': proba_pos,
        'NEG_pred': pred_neg,
        'POS_pred': pred_pos,
        'proba_diff': proba_neg - proba_pos,
        'proba_avg': (proba_neg + proba_pos) / 2
    })
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_stack, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Configuration validation croisée
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # Test de différents méta-modèles
    stacking_results = {}
    best_auc = 0
    best_meta_model = None
    best_meta_name = ""
    
    for meta_name, meta_model in META_MODELS.items():
        print(f"\n{'='*50}")
        print(f"Testing meta-model: {meta_name}")
        print(f"{'='*50}")
        
        # Pipeline avec SMOTE et méta-modèle
        meta_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', meta_model)
        ])
        
        # Optimisation des hyperparamètres
        grid = RandomizedSearchCV(
            meta_pipe,
            META_PARAMS[meta_name],
            n_iter=15,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=3,
            random_state=42
        )
        grid.fit(X_train, y_train)
        best_meta = grid.best_estimator_
        print(f"Best params for {meta_name}: {grid.best_params_}")
        
        # Évaluation
        test_metrics = save_evaluation_metrics(
            best_meta, X_test, y_test, 
            f"results/classif_{classification_id}/stacking", 
            f'stacking_{meta_name}'
        )
        stacking_results[meta_name] = test_metrics
        
        # Sauvegarde du modèle intermédiaire
        joblib.dump(best_meta, os.path.join(model_dir, f"stacking_{meta_name}.pkl"))
        
        # Mise à jour du meilleur méta-modèle
        if test_metrics['roc_auc'] > best_auc:
            best_auc = test_metrics['roc_auc']
            best_meta_model = best_meta
            best_meta_name = meta_name
    
    # Sauvegarde du meilleur modèle de stacking
    joblib.dump(best_meta_model, os.path.join(model_dir, "best_stacking_model.pkl"))
    
    # Rapport final
    print("\nStacking Model Comparison:")
    print(pd.DataFrame(stacking_results).T[['roc_auc', 'accuracy', 'f1_weighted']])
    
    print(f"\nBest stacking model: {best_meta_name} with AUC: {best_auc:.4f}")
    
    # Sauvegarde des prédictions du meilleur modèle
    test_results = pd.DataFrame({
        'NEG_proba': X_test['NEG_proba'],
        'POS_proba': X_test['POS_proba'],
        'true_label': y_test,
        'predicted_label': best_meta_model.predict(X_test),
        'predicted_proba': best_meta_model.predict_proba(X_test)[:, 1]
    })
    test_results.to_csv(
        f"results/classif_{classification_id}/stacking/best_stacking_predictions.csv", 
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train stacking model for clinical deficit classification')
    parser.add_argument('--classification', type=int, required=True, 
                        choices=range(1, 11), help='Clinical deficit ID (1-10)')
    args = parser.parse_args()
    
    main(args.classification)