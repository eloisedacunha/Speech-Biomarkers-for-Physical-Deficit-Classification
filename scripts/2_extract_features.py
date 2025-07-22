import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from features.acoustic_features import extract_acoustic
from features.temporal_features import extract_temporal
from features.linguistic_features import extract_linguistic
from utils.config import PHONE_CATEGORIES, SEMANTIC_CATEGORIES

def process_condition(base_path):
    """Extract features for a specific condition"""
    features = {}
    
    # Acoustic features from audio
    audio_path = f"{base_path}.wav"
    acoustic_features = extract_acoustic(audio_path)
    features.update(acoustic_features)
    
    # Temporal features from TextGrid
    textgrid_path = f"{base_path}.TextGrid"
    temporal_features = extract_temporal(textgrid_path, PHONE_CATEGORIES)
    features.update(temporal_features)
    
    # Linguistic features from transcript
    transcript_path = f"{base_path}.txt"
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        linguistic_features = extract_linguistic(transcript, SEMANTIC_CATEGORIES)
        features.update(linguistic_features)
    except Exception as e:
        print(f"Error reading transcript: {str(e)}")
    
    return features

def extract_all_features(data_dir="data/raw", output_path="data/processed/all_speech_features.csv"):
    """Main function to extract features for all participants and conditions"""
    participants = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    all_features = []
    
    for participant in tqdm(participants, desc="Processing participants"):
        participant_dir = os.path.join(data_dir, participant)
        features = {"participant_id": participant}
        
        # Process both conditions
        for condition in ["positive", "negative"]:
            base_path = os.path.join(participant_dir, condition)
            
            try:
                cond_features = process_condition(base_path)
                features.update({f"{condition}_{k}": v for k, v in cond_features.items()})
            except Exception as e:
                print(f"Error processing {condition} for {participant}: {str(e)}")
        
        all_features.append(features)
    
    # Create and save DataFrame
    df = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully extracted features for {len(participants)} participants")
    
    return df

if __name__ == "__main__":
    extract_all_features()