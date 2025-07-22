import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.config import DEFICIT_MAP, CLINICAL_DEFICITS

def prepare_classification_data(deficit_id):
    """Prépare les données de classification pour un déficit spécifique"""
    print(f"Préparation des données pour le déficit {deficit_id}: {DEFICIT_MAP[deficit_id]}")
    
    # Charger les caractéristiques combinées
    features_df = pd.read_csv("data/processed/all_speech_features.csv")
    
    # Charger les étiquettes cliniques
    clinical_df = pd.read_csv("data/processed/clinical_labels.csv")
    
    # Fusionner les données
    merged_df = pd.merge(features_df, clinical_df, on="participant_id")
    
    # Filtrer pour le déficit spécifique
    deficit_col = f"deficit_{deficit_id}"
    if deficit_col not in merged_df.columns:
        raise ValueError(f"Colonne {deficit_col} non trouvée dans les données cliniques")
    
    # Préparer les données pour NEG et POS
    for emotion in ["NEG", "POS"]:
        # Filtrer par émotion
        emotion_df = merged_df[merged_df["emotion"] == emotion].copy()
        
        # Séparer les caractéristiques et la cible
        X = emotion_df.drop(columns=["participant_id", "audio_file", "emotion", "transcription"] + CLINICAL_DEFICITS)
        y = emotion_df[deficit_col].map({1: "deficit", 0: "norm"})
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Ajouter la cible
        final_df = X_scaled.copy()
        final_df["deficit_class"] = y
        
        # Sauvegarder
        output_path = f"data/classifications/classification_{deficit_id}_{emotion}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Données sauvegardées: {output_path} - {final_df.shape[0]} échantillons")

def main():
    # Créer le dossier classifications
    os.makedirs("data/classifications", exist_ok=True)
    
    # Préparer les données pour chaque déficit
    for deficit_id in DEFICIT_MAP.keys():
        prepare_classification_data(deficit_id)
    
    print("Toutes les classifications ont été préparées avec succès!")

if __name__ == "__main__":
    main()