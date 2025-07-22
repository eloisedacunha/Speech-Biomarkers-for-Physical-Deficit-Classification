import textgrid
import numpy as np
from collections import defaultdict

def extract_temporal_features(textgrid_path, phone_categories):
    """Extract temporal speech features from TextGrid"""
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        
        # Find phone and word tiers
        phone_tier = next((t for t in tg if t.name.lower() == "phones"), None)
        word_tier = next((t for t in tg if t.name.lower() == "words"), None)
        
        if phone_tier is None or word_tier is None:
            raise ValueError("TextGrid must have 'phones' and 'words' tiers")
        
        # Phone analysis
        phone_durations = []
        category_counts = defaultdict(int)
        vowel_durations = []
        
        for interval in phone_tier:
            phone = interval.mark.lower().strip()
            duration = interval.duration()
            
            if phone in phone_categories['silences']:
                continue
                
            phone_durations.append(duration)
            
            # Categorize phone
            for cat, phones in phone_categories.items():
                if cat == 'silences':
                    continue
                if phone in phones:
                    category_counts[cat] += 1
                    if cat == 'vowels':
                        vowel_durations.append(duration)
                    break
        
        # Word and pause analysis
        word_durations = []
        pause_durations = []
        word_phoneme_counts = []
        
        for interval in word_tier:
            text = interval.mark.strip()
            duration = interval.duration()
            
            if not text or text in phone_categories['silences']:
                pause_durations.append(duration)
            else:
                word_durations.append(duration)
                
                # Count phonemes in this word
                count = 0
                for phone_interval in phone_tier:
                    if (phone_interval.minTime >= interval.minTime and 
                        phone_interval.maxTime <= interval.maxTime and
                        phone_interval.mark.strip().lower() not in phone_categories['silences']):
                        count += 1
                word_phoneme_counts.append(count)
        
        # Calculate features
        total_phone_time = sum(phone_durations)
        total_pause_time = sum(pause_durations)
        
        return {
            'total_speech_duration': total_phone_time,
            'total_words': len(word_durations),
            'speech_rate': len(word_durations) / total_phone_time if total_phone_time > 0 else 0,
            'articulation_rate': len(phone_durations) / total_phone_time if total_phone_time > 0 else 0,
            'mean_word_duration': np.mean(word_durations) if word_durations else 0,
            'mean_word_length': np.mean(word_phoneme_counts) if word_phoneme_counts else 0,
            'number_of_pauses': len(pause_durations),
            'total_pause_duration': total_pause_time,
            'mean_pause_duration': np.mean(pause_durations) if pause_durations else 0,
            'vowel_ratio': category_counts['vowels'] / len(phone_durations) if phone_durations else 0,
            'plosive_ratio': category_counts['plosives'] / len(phone_durations) if phone_durations else 0,
            'fricative_ratio': category_counts['fricatives'] / len(phone_durations) if phone_durations else 0,
            'nasal_ratio': category_counts['nasals'] / len(phone_durations) if phone_durations else 0,
            'liquid_ratio': category_counts['liquids'] / len(phone_durations) if phone_durations else 0,
            'glide_ratio': category_counts['glides'] / len(phone_durations) if phone_durations else 0,
            'mean_vowel_duration': np.mean(vowel_durations) if vowel_durations else 0
        }
    except Exception as e:
        print(f"Error extracting temporal features from {textgrid_path}: {str(e)}")
        return {}