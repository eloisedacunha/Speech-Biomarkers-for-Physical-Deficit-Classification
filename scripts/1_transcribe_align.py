import os
import subprocess
import whisper
import textgrid
from tqdm import tqdm
from utils.config import PHONE_CATEGORIES

def transcribe_audio(audio_path, model):
    """Transcrit l'audio avec Whisper"""
    result = model.transcribe(audio_path)
    return result["text"]

def align_audio(audio_path, transcript_path):
    """Aligne l'audio avec MFA"""
    base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    textgrid_path = os.path.join(os.path.dirname(audio_path), f"{base_name}.TextGrid")
    
    cmd = [
        "mfa", "align",
        audio_path,
        transcript_path,
        "french",
        textgrid_path,
        "--clean",
        "--single_speaker"
    ]
    subprocess.run(cmd, check=True)
    return textgrid_path

def process_participant(participant_dir):
    """Traite un participant complet"""
    results = {}
    model = whisper.load_model("medium")
    
    for condition in ["positive", "negative"]:
        base_path = os.path.join(participant_dir, condition)
        audio_path = f"{base_path}.wav"
        
        # Transcription
        transcript = transcribe_audio(audio_path, model)
        transcript_path = f"{base_path}.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Alignement
        textgrid_path = align_audio(audio_path, transcript_path)
        
        results[condition] = {
            'audio': audio_path,
            'transcript': transcript_path,
            'textgrid': textgrid_path
        }
    
    return results

def main(data_dir="data/raw"):
    participants = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    for participant in tqdm(participants, desc="Processing participants"):
        participant_dir = os.path.join(data_dir, participant)
        process_participant(participant_dir)

if __name__ == "__main__":
    main()