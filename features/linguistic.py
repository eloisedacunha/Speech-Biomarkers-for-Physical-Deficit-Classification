import spacy
import re
from collections import Counter

nlp = spacy.load("fr_core_news_sm")

def extract_linguistic_features(transcript, semantic_categories):
    """Extract linguistic features from transcript"""
    try:
        # Clean transcript
        transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove annotations
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        if not transcript:
            return {}
        
        # Process with spaCy
        doc = nlp(transcript)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        
        total_words = len(words)
        if total_words == 0:
            return {}
        
        # Part-of-speech counts
        pos_counts = Counter([token.pos_ for token in doc])
        
        # Semantic category counts
        semantic_counts = {category: 0 for category in semantic_categories.keys()}
        negated_words = set()
        
        # Detect negations
        for token in doc:
            if token.dep_ == "neg" and token.head.i < len(doc) - 1:
                next_token = doc[token.head.i + 1]
                negated_words.add(next_token.lemma_.lower())
        
        # Count words in semantic categories
        for token in doc:
            lemma = token.lemma_.lower()
            for category, word_list in semantic_categories.items():
                if lemma in word_list:
                    # Handle negation for emotions
                    if category in ['pos_emotion', 'neg_emotion']:
                        if lemma in negated_words:
                            # Flip emotion if negated
                            if category == 'pos_emotion':
                                semantic_counts['neg_emotion'] += 1
                            else:
                                semantic_counts['pos_emotion'] += 1
                        else:
                            semantic_counts[category] += 1
                    else:
                        semantic_counts[category] += 1
        
        # Calculate ratios
        features = {
            'verb_ratio': pos_counts.get('VERB', 0) / total_words,
            'noun_ratio': pos_counts.get('NOUN', 0) / total_words,
            'pronoun_ratio': pos_counts.get('PRON', 0) / total_words,
            'adj_ratio': pos_counts.get('ADJ', 0) / total_words,
            'adv_ratio': pos_counts.get('ADV', 0) / total_words,
            'time_ratio': semantic_counts['time'] / total_words,
            'narrator_ratio': semantic_counts['narrator'] / total_words,
            'pos_emotion_ratio': semantic_counts['pos_emotion'] / total_words,
            'neg_emotion_ratio': semantic_counts['neg_emotion'] / total_words,
            'location_ratio': semantic_counts['location'] / total_words,
            'lexical_diversity': len(set(words)) / total_words,  # TTR
            'mean_sentence_length': total_words / len(sentences) if sentences else 0,
            'hesitation_ratio': sum(1 for word in words if word.lower() in ['euh', 'hum', 'ah']) / total_words
        }
        return features
    except Exception as e:
        print(f"Error extracting linguistic features: {str(e)}")
        return {}