import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

def extract_acoustic_features(audio_path):
    """Extract acoustic features from audio file"""
    try:
        # Load audio with librosa (mono, 16kHz)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract fundamental frequency (F0) with WORLD
        f0, t = pw.dio(y.astype(np.float64), sr)
        f0 = pw.stonemask(y.astype(np.float64), f0, t, sr)
        valid_f0 = f0[f0 > 0]
        
        # Calculate pitch statistics
        pitch_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 0
        pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
        pitch_range = np.max(valid_f0) - np.min(valid_f0) if len(valid_f0) > 0 else 0
        
        # Extract intensity (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        intensity_mean = np.mean(rms)
        intensity_std = np.std(rms)
        
        # Extract harmonics-to-noise ratio (HNR) using WORLD
        _, sp, _ = pw.wav2world(y.astype(np.float64), sr)
        hnr = 10 * np.log10(np.mean(sp, axis=1) / np.var(sp, axis=1))
        hnr_mean = np.mean(hnr)
        
        # Jitter and shimmer (simplified)
        jitter = np.mean(np.abs(np.diff(valid_f0))) / pitch_mean if pitch_mean > 0 else 0
        shimmer = np.mean(np.abs(np.diff(rms))) / intensity_mean if intensity_mean > 0 else 0
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # Formants (simplified: using spectral peaks)
        formant1 = []
        formant2 = []
        for frame in sp.T:
            peaks = np.argsort(frame)[::-1][:2]
            formant1.append(peaks[0])
            formant2.append(peaks[1])
        
        formant1_mean = np.mean(formant1) if formant1 else 0
        formant2_mean = np.mean(formant2) if formant2 else 0
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'hnr': hnr_mean,
            'jitter': jitter,
            'shimmer': shimmer,
            'spectral_centroid': spectral_centroid_mean,
            'formant1_mean': formant1_mean,
            'formant2_mean': formant2_mean,
            'zcr_mean': zcr_mean
        }
    except Exception as e:
        print(f"Error extracting acoustic features from {audio_path}: {str(e)}")
        return {}