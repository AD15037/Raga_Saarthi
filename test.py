import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require a GUI
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from flask_cors import CORS
import os
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import uuid
import pandas as pd
from difflib import SequenceMatcher
import tensorflow as tf
import traceback
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from flask import send_from_directory
import json
from datetime import datetime
import joblib
from speech_to_text import speech_to_text   # Import the speech-to-text function from "speech_to_text.py" file
import logging
import crepe
from dtw import dtw

# New imports for user management and ML personalization
from flask_bcrypt import Bcrypt
from flask_session import Session
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import shutil

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_url_path='/static', static_folder='static')
bcrypt = Bcrypt(app)

# Configure server-side sessions
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "raag_saarthi_secret_key")
Session(app)

CORS(app, resources={r"/": {"origins": ""}}, supports_credentials=True)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)
os.makedirs("user_data", exist_ok=True)
os.makedirs("models/personalization", exist_ok=True)

# User data storage
USER_DATA_PATH = "user_data"

def get_user_data_path(username):
    """Get the path for a user's data directory"""
    user_dir = os.path.join(USER_DATA_PATH, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def load_user_profile(username):
    """Load user profile or create a new one if it doesn't exist"""
    profile_path = os.path.join(get_user_data_path(username), "profile.json")
    
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            return json.load(f)
    else:
        # Create new user profile with default values
        default_profile = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "practice_sessions": 0,
            "total_practice_time": 0,
            "ragas_practiced": [],
            "skill_level": "beginner",
            "vocal_range": {
                "min": None,
                "max": None
            },
            "preferred_ragas": [],
            "achievements": [],
            "practice_streak": 0,
            "last_practice": None,
            "personalization_data": {
                "pitch_accuracy": 0.0,
                "rhythm_stability": 0.0,
                "gamaka_proficiency": 0.0,
                "breath_control": 0.0
            }
        }
        save_user_profile(username, default_profile)
        return default_profile

def save_user_profile(username, profile_data):
    """Save user profile to disk"""
    profile_path = os.path.join(get_user_data_path(username), "profile.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)

def save_performance_record(username, performance_data):
    """Save a new performance record for the user"""
    user_dir = get_user_data_path(username)
    performances_path = os.path.join(user_dir, "performances.json")
    
    # Load existing performances or create new list
    if os.path.exists(performances_path):
        with open(performances_path, "r") as f:
            performances = json.load(f)
    else:
        performances = []
    
    # Add timestamp to performance data
    performance_data["timestamp"] = datetime.now().isoformat()
    
    # Add to performances list
    performances.append(performance_data)
    
    # Save updated performances
    with open(performances_path, "w") as f:
        json.dump(performances, f, indent=2)
    
    return len(performances)  # Return total number of performances

def analyze_vocal_characteristics(audio_data, sr):
    """
    Analyze vocal characteristics including range, timbre, and stability
    Returns a dictionary of vocal characteristics
    """
    # Extract pitch
    pitch = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), 
                       fmax=librosa.note_to_hz('C7'),
                       sr=sr)
    
    # Filter out zero or invalid pitch values
    valid_pitch = pitch[pitch > 0]
    
    if len(valid_pitch) == 0:
        return {
            "vocal_range": {"min": 0, "max": 0, "mean": 0},
            "timbre": {"brightness": 0, "roughness": 0},
            "stability": {"pitch_stability": 0, "vibrato_rate": 0},
            "breath_control": 0
        }
    
    # Calculate vocal range in Hz
    min_pitch = np.min(valid_pitch)
    max_pitch = np.max(valid_pitch)
    mean_pitch = np.mean(valid_pitch)
    
    # Convert to note names for better readability
    min_note = librosa.hz_to_note(min_pitch)
    max_note = librosa.hz_to_note(max_pitch)
    mean_note = librosa.hz_to_note(mean_pitch)
    
    # Calculate spectral characteristics for timbre
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0])
    
    # Get full spectral contrast for roughness calculation
    spectral_contrast_full = librosa.feature.spectral_contrast(y=audio_data, sr=sr)[0]
    
    # Brightness is related to spectral centroid
    brightness = spectral_centroid / (sr/2)  # Normalize by Nyquist frequency
    
    # Roughness can be estimated from spectral contrast
    if len(spectral_contrast_full) > 1:
        roughness = np.mean(np.abs(np.diff(spectral_contrast_full)))
    else:
        roughness = 0.0
    
    # Calculate pitch stability
    pitch_stability = 1.0 - (np.std(valid_pitch) / mean_pitch)
    
    # Detect vibrato
    # Get pitch derivative and smooth it
    if len(valid_pitch) > 1:
        pitch_diff = np.diff(valid_pitch)
        pitch_diff_smooth = gaussian_filter1d(pitch_diff, sigma=2)
        
        # Count zero crossings to estimate vibrato rate
        zero_crossings = np.sum(np.diff(np.signbit(pitch_diff_smooth)))
        vibrato_rate = zero_crossings / (len(pitch_diff_smooth) / sr)
    else:
        vibrato_rate = 0
    
    # Estimate breath control based on amplitude envelope variation
    amplitude_envelope = librosa.feature.rms(y=audio_data)[0]
    if len(amplitude_envelope) > 0:
        breath_control = 1.0 - min(1.0, np.std(amplitude_envelope) / np.mean(amplitude_envelope))
    else:
        breath_control = 0
        
    return {
        "vocal_range": {
            "min_hz": float(min_pitch),
            "max_hz": float(max_pitch),
            "mean_hz": float(mean_pitch),
            "min_note": min_note,
            "max_note": max_note,
            "mean_note": mean_note
        },
        "timbre": {
            "brightness": float(brightness),
            "roughness": float(roughness),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth)
        },
        "stability": {
            "pitch_stability": float(pitch_stability),
            "vibrato_rate": float(vibrato_rate)
        },
        "breath_control": float(breath_control)
    }

def evaluate_performance(audio_data, sr, raga_info, transcribed_text=None):
    """
    Evaluate a user's performance against raga characteristics
    Returns metrics and feedback
    """
    # Extract pitch and convert to swaras
    pitch = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), 
                       fmax=librosa.note_to_hz('C7'),
                       sr=sr)
    
    detected_swaras = improved_pitch_to_swara(pitch, sr)
    
    # Get reference patterns
    aaroh_avroh = raga_info.get('aaroh_-_avroh', '')
    aaroh_parts = aaroh_avroh.split('-') if '-' in aaroh_avroh else [aaroh_avroh, ""]
    aaroh = tokenize_swaras(aaroh_parts[0])
    avroh = tokenize_swaras(aaroh_parts[1] if len(aaroh_parts) > 1 else "")
    pakad = tokenize_swaras(raga_info.get('pakad', ''))
    
    # Calculate adherence to raga structure
    aaroh_adherence = 0
    avroh_adherence = 0
    pakad_adherence = 0
    
    # Check for aaroh patterns
    for i in range(len(detected_swaras) - len(aaroh) + 1):
        subsequence = detected_swaras[i:i+len(aaroh)]
        similarity = match_score(subsequence, aaroh)
        aaroh_adherence = max(aaroh_adherence, similarity)
    
    # Check for avroh patterns
    if len(avroh) > 0:
        for i in range(len(detected_swaras) - len(avroh) + 1):
            subsequence = detected_swaras[i:i+len(avroh)]
            similarity = match_score(subsequence, avroh)
            avroh_adherence = max(avroh_adherence, similarity)
    
    # Check for pakad patterns
    if len(pakad) > 0:
        for i in range(len(detected_swaras) - len(pakad) + 1):
            subsequence = detected_swaras[i:i+len(pakad)]
            similarity = match_score(subsequence, pakad)
            pakad_adherence = max(pakad_adherence, similarity)
    
    # Analyze pitch accuracy
    gamaka_features = detect_gamakas(pitch, sr)
    
    # Detect vadi-samvadi in performance
    vadi_samvadi_performed = detect_vadi_samvadi(pitch, sr)
    expected_vadi = raga_info.get('vadi', '')
    expected_samvadi = raga_info.get('samvadi', '')
    
    vadi_samvadi_accuracy = 0
    if expected_vadi and expected_samvadi:
        if vadi_samvadi_performed[0] == expected_vadi:
            vadi_samvadi_accuracy += 50
        if vadi_samvadi_performed[1] == expected_samvadi:
            vadi_samvadi_accuracy += 50
    
    # Calculate rhythm stability if possible
    rhythm_stability = 0.0
    try:
        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onsets) > 1:
            # Calculate inter-onset intervals
            iois = np.diff(librosa.frames_to_time(onsets, sr=sr))
            # Rhythm stability is inversely related to IOI variance
            rhythm_stability = 100.0 - min(100.0, (np.std(iois) / np.mean(iois)) * 100.0)
    except:
        pass
    
    # Calculate overall performance score
    aaroh_avroh_score = (aaroh_adherence + avroh_adherence) / 2 if avroh else aaroh_adherence
    structure_score = 0.5 * aaroh_avroh_score + 0.5 * pakad_adherence if pakad else aaroh_avroh_score
    
    # Overall score combines structure adherence, vadi/samvadi accuracy and rhythm stability
    overall_score = 0.6 * structure_score + 0.2 * vadi_samvadi_accuracy + 0.2 * rhythm_stability
    
    # Generate feedback based on performance
    feedback = []
    
    # Structure feedback
    if aaroh_adherence < 60:
        feedback.append({
            "type": "structure",
            "area": "aaroh",
            "message": "Work on the ascending pattern of this raga. Try practicing the aaroh slowly and clearly."
        })
    
    if avroh and avroh_adherence < 60:
        feedback.append({
            "type": "structure",
            "area": "avroh",
            "message": "Focus on the descending pattern of this raga. Practice the avroh to improve precision."
        })
    
    if pakad and pakad_adherence < 60:
        feedback.append({
            "type": "structure",
            "area": "pakad",
            "message": f"The pakad (characteristic phrase) wasn't clearly evident. Practice this essential pattern: {' '.join(pakad)}"
        })
    
    # Vadi/Samvadi feedback
    if expected_vadi and vadi_samvadi_performed[0] != expected_vadi:
        feedback.append({
            "type": "emphasis",
            "area": "vadi",
            "message": f"Try emphasizing the vadi note ({expected_vadi}) more in your performance."
        })
    
    # Rhythm feedback
    if rhythm_stability < 70:
        feedback.append({
            "type": "rhythm",
            "area": "stability",
            "message": "Work on maintaining consistent rhythm throughout your performance."
        })
    
    # Analyze pronunciation if transcribed_text is available
    pronunciation_score = 0.0
    if transcribed_text:
        # Simple analysis of lyrics presence
        # A more sophisticated analysis would compare with a reference
        words = transcribed_text.strip().split()
        if len(words) > 5:
            pronunciation_score = 70.0 + min(30.0, len(words) * 0.5)
        else:
            pronunciation_score = max(40.0, len(words) * 10.0)
        
        feedback.append({
            "type": "pronunciation",
            "area": "lyrics",
            "message": "Continue practicing clear pronunciation of lyrics while maintaining the raga structure."
        })
    
    # Return evaluation results
    return {
        "overall_score": float(overall_score),
        "structure_adherence": {
            "aaroh": float(aaroh_adherence),
            "avroh": float(avroh_adherence),
            "pakad": float(pakad_adherence)
        },
        "vadi_samvadi_accuracy": float(vadi_samvadi_accuracy),
        "rhythm_stability": float(rhythm_stability),
        "pronunciation_score": float(pronunciation_score) if transcribed_text else None,
        "detected_patterns": {
            "vadi_samvadi": vadi_samvadi_performed,
            "gamaka_features": [float(x) for x in gamaka_features]
        },
        "feedback": feedback
    }

# Load trained CNN model
MODEL_PATH = "models/cnn_raga_classifier.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load the label encoder
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load feature extraction parameters
    with open("models/feature_params.pkl", "rb") as f:
        feature_params = pickle.load(f)
        n_mels = feature_params.get("n_mels", 128)
        n_chroma = feature_params.get("n_chroma", 12)
        n_contrast = feature_params.get("n_contrast", 7)
        max_time_steps = feature_params.get("max_time_steps", 109)
    
    # Load scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Set model_loaded flag
    model_loaded = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model_loaded = False

# Load raga rules CSV if it exists
RAGA_RULES_PATH = "Dataset.csv"
try:
    raga_df = pd.read_csv(RAGA_RULES_PATH, encoding="latin-1")
    raga_df.columns = [col.strip().lower().replace(" ", "_") for col in raga_df.columns]  # normalize column names
    raga_rules_loaded = True
except Exception as e:
    print(f"Failed to load raga rules: {e}")
    raga_rules_loaded = False

# Define Indian classical music notes mapping
# This enhanced mapping accounts for microtones and variations
SWARA_MAPPING = {
    'C': 'S',   # Shadja
    'C#': 'r',  # Komal Rishabh
    'Db': 'r',  # Komal Rishabh (alternate notation)
    'D': 'R',   # Shuddha Rishabh
    'D#': 'g',  # Komal Gandhar
    'Eb': 'g',  # Komal Gandhar (alternate notation)
    'E': 'G',   # Shuddha Gandhar
    'F': 'm',   # Komal Madhyam
    'F#': 'M',  # Shuddha Madhyam
    'Gb': 'M',  # Shuddha Madhyam (alternate notation)
    'G': 'P',   # Pancham
    'G#': 'd',  # Komal Dhaivat
    'Ab': 'd',  # Komal Dhaivat (alternate notation)
    'A': 'D',   # Shuddha Dhaivat
    'A#': 'n',  # Komal Nishad
    'Bb': 'n',  # Komal Nishad (alternate notation)
    'B': 'N',   # Shuddha Nishad
}

# Preprocessing functions
def remove_silence(audio, sr, top_db=20):
    try:
        non_silent = librosa.effects.split(audio, top_db=top_db)
        audio_clean = np.concatenate([audio[start:end] for start, end in non_silent]) if non_silent.size else audio
        return audio_clean
    except Exception as e:
        logging.error(f"Error removing silence: {e}")
        return audio

def normalize_audio(audio):
    max_abs = np.max(np.abs(audio))
    return audio / max_abs if max_abs != 0 else audio

# Feature extraction function - Updated to match training_model.py
def features_extractor(file, n_mels=128, n_chroma=12, n_contrast=7, max_time_steps=109):
    try:
        audio, sr = librosa.load(file, sr=None, duration=30)
        audio = remove_silence(audio, sr)
        audio = normalize_audio(audio)
        
        # Log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
        
        # Temporal features (onset strength)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_features = np.array([np.mean(onset_env), np.std(onset_env)])
        onset_features_2d = np.repeat(onset_features[:, np.newaxis], max_time_steps, axis=1)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=n_contrast-1)
        
        # Pad or truncate to fixed length
        for spec in [log_mel_spec, chroma, onset_features_2d, spectral_contrast]:
            if spec.shape[1] < max_time_steps:
                pad_width = max_time_steps - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
            else:
                spec = spec[:, :max_time_steps]
        
        # Concatenate features
        combined_spec = np.concatenate([log_mel_spec, chroma, onset_features_2d, spectral_contrast], axis=0)
        
        # Normalize features
        combined_spec_reshaped = combined_spec.reshape(1, -1)
        combined_spec_normalized = scaler.transform(combined_spec_reshaped).reshape(combined_spec.shape)
        
        logging.debug(f"Extracted features shape: {combined_spec_normalized.shape}")
        return combined_spec_normalized[:, :, np.newaxis]
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None

# Segment-based feature extraction
def segment_features(file, segment_length=5):
    try:
        audio, sr = librosa.load(file, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        segments = []
        for start in np.arange(0, duration, segment_length):
            end = min(start + segment_length, duration)
            segment = audio[int(start * sr):int(end * sr)]
            segments.append((segment, sr))
        
        all_features = []
        for segment, sr in segments:
            temp_path = os.path.join("temp", f"segment_{uuid.uuid4().hex}.wav")
            os.makedirs("temp", exist_ok=True)
            librosa.output.write_wav(temp_path, segment, sr)
            feature = features_extractor(temp_path)
            if feature is not None:
                all_features.append(feature)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return np.array(all_features) if all_features else None
    except Exception as e:
        logging.error(f"Error in segment_features: {e}")
        return None

# Pitch detection using CREPE
def get_pitch(audio, sr):
    try:
        time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True)
        return frequency[confidence > 0.5]
    except Exception as e:
        logging.error(f"Error in pitch detection: {e}")
        return np.array([])

# Utility: extract swara tokens from a pattern string
def tokenize_swaras(swara_string):
    return [note.strip() for note in swara_string.replace('-', ' ').replace(',', ' ').split() if note.strip()]

# Utility: compare swara sequences
def match_score(input_seq, rule_seq):
    return SequenceMatcher(None, input_seq, rule_seq).ratio() * 100

# Rule-based matching with DTW
def match_score_dtw(input_seq, rule_seq):
    try:
        # Convert sequences to numerical values for DTW
        swara_to_num = {swara: idx for idx, swara in enumerate(set(input_seq + rule_seq))}
        input_num = [swara_to_num[swara] for swara in input_seq]
        rule_num = [swara_to_num[swara] for swara in rule_seq]
        distance, _ = dtw(input_num, rule_num, dist=lambda x, y: abs(x - y))
        return 100 * (1 - distance / max(len(input_seq), len(rule_seq)))
    except Exception:
        return match_score(input_seq, rule_seq)  # Fallback to SequenceMatcher

# Create pitch histogram with 24 bins
def create_pitch_histogram(pitch_values, sr=22050, bins=24):
    valid_pitch = pitch_values[pitch_values > 0]
    if len(valid_pitch) == 0:
        return np.zeros(bins)
    cents = 1200 * np.log2(valid_pitch / 261.63)
    pitch_classes = (cents / 50) % 24  # 50 cents per bin
    histogram, _ = np.histogram(pitch_classes, bins=bins, range=(0, 24), density=True)
    return histogram / np.sum(histogram) if histogram.sum() != 0 else histogram

# Improved: Match raga based on pakad, aaroh-avroh, and characteristic phrases
def find_best_matching_raga(pitch_notes, pitch_histogram=None, vadi_samvadi=None, gamaka_features=None):
    if not raga_rules_loaded:
        return "Unknown", 0, {}
    
    matching_scores = {}
    best_raga = None
    best_score = 0
    best_details = {}

    for _, row in raga_df.iterrows():
        raga_name = row['name_of_the_raag']
        aaroh_avroh = row.get('aaroh_-_avroh', '')
        pakad = row.get('pakad', '')

        # Enhanced sequence matching with DTW
        rule_tokens = tokenize_swaras(f"{aaroh_avroh} {pakad}")
        sequence_score = match_score_dtw(pitch_notes, rule_tokens)
        
        # Pakad-specific matching
        pakad_tokens = tokenize_swaras(pakad)
        pakad_score = match_score_dtw(pitch_notes, pakad_tokens) if pakad_tokens else sequence_score
        
        final_score = 0.4 * sequence_score + 0.4 * pakad_score
        score_details = {"sequence_match": sequence_score, "pakad_match": pakad_score}
        
        # Pitch histogram matching
        if pitch_histogram is not None and 'pitch_histogram' in row:
            try:
                raga_histogram = np.array([float(x) for x in row['pitch_histogram'].split(',')])
                histogram_score = np.corrcoef(pitch_histogram, raga_histogram)[0, 1] * 100
                final_score += 0.15 * histogram_score
                score_details["histogram_match"] = histogram_score
            except (ValueError, KeyError, IndexError):
                pass
        
        # Vadi-samvadi matching
        if vadi_samvadi is not None and 'vadi' in row and 'samvadi' in row:
            try:
                vadi_match = 100 if vadi_samvadi[0] == row['vadi'] else 0
                samvadi_match = 100 if vadi_samvadi[1] == row['samvadi'] else 0
                vadi_score = 0.7 * vadi_match + 0.3 * samvadi_match
                final_score += 0.05 * vadi_score
                score_details["vadi_samvadi_match"] = vadi_score
            except (KeyError, IndexError):
                pass
        
        # Gamaka features matching
        if gamaka_features is not None and 'gamaka_features' in row:
            try:
                raga_gamaka = np.array([float(x) for x in row['gamaka_features'].split(',')])
                gamaka_score = np.corrcoef(gamaka_features, raga_gamaka)[0, 1] * 100
                final_score = 0.8 * final_score + 0.2 * gamaka_score
                score_details["gamaka_match"] = gamaka_score
            except (ValueError, KeyError, IndexError):
                pass
        
        matching_scores[raga_name] = final_score
        
        if final_score > best_score:
            best_score = final_score
            best_raga = raga_name
            best_details = score_details

    top_ragas = sorted(matching_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return best_raga, best_score, {"top_matches": top_ragas, "details": best_details}

# Hybrid prediction
def hybrid_predict(cnn_probs, rule_scores):
    combined_scores = 0.7 * cnn_probs + 0.3 * rule_scores
    return np.argmax(combined_scores)

# Contextual filtering
def filter_ragas_by_vocal_range(raga_df, vocal_range):
    min_hz = vocal_range.get("min_hz", 100)
    max_hz = vocal_range.get("max_hz", 1000)
    return raga_df[raga_df["swara_set"].apply(lambda x: is_within_vocal_range(x, min_hz, max_hz))]

def is_within_vocal_range(swara_set, min_hz, max_hz):
    # Placeholder: Convert swara_set to frequencies and check range
    return True  # Implement based on swara-to-frequency mapping

# NEW: Enhanced pitch to note conversion with better microtone handling
def improved_pitch_to_swara(pitch_values, sr=22050):
    """
    Convert pitch values to Indian classical music swaras with microtone detection.
    """
    if len(pitch_values) == 0 or np.all(pitch_values <= 0):
        return []
    
    # Filter out zero or negative values
    valid_pitch = pitch_values[pitch_values > 0]
    
    if len(valid_pitch) == 0:
        return []
    
    # Convert from Hz to cents relative to C4 (middle C)
    C4_FREQ = 261.63  # Frequency of C4 in Hz
    cents = 1200 * np.log2(valid_pitch / C4_FREQ)
    
    # Map cents to notes
    notes = []
    for cent in cents:
        # Round to nearest semitone (100 cents)
        semitone = round(cent / 100)
        
        # Get the note name based on semitone distance from C
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = semitone % 12
        if note_index < 0:
            note_index += 12
        
        western_note = note_names[note_index]
        
        # Convert to Indian swara
        if western_note in SWARA_MAPPING:
            notes.append(SWARA_MAPPING[western_note])
    
    return notes

def get_notation_guide():
    """Return a dictionary with explanations of Indian classical music notation symbols and dataset columns"""
    return {
        "notation_symbols": {
            "notes": {
                "S": "Shadj (Sa) - The tonic or base note",
                "R/r": "Rishabh (Re) - Second note (R: natural/shuddha, r: flat/komal)",
                "G/g": "Gandhar (Ga) - Third note (G: natural/shuddha, g: flat/komal)",
                "M/m": "Madhyam (Ma) - Fourth note (M: sharp/tivra, m: natural/shuddha)",
                "P": "Pancham (Pa) - Fifth note (always natural)",
                "D/d": "Dhaivat (Dha) - Sixth note (D: natural/shuddha, d: flat/komal)",
                "N/n": "Nishad (Ni) - Seventh note (N: natural/shuddha, n: flat/komal)"
            },
            "octave_indicators": {
                ",X": "Note X in lower octave (e.g., ,N is lower Ni)",
                "X'": "Note X in upper octave (e.g., S' is upper Sa)",
                "X''": "Note X in second upper octave"
            },
            "pattern_symbols": {
                "-": "Separator between ascending (Aaroh) and descending (Avroh) patterns",
                ";": "Phrase separator within a pattern",
                "~": "Gamak - oscillation or ornamentation on a note",
                "()": "Notes played with special emphasis or technique (e.g., (P))",
                " ": "Space between notes indicates separate notes",
                "X X": "Notes played separately",
                "XX": "Notes played in quick succession"
            }
        },
        "dataset_columns": {
            "Name of the raag": "The formal name of the raga",
            "Aaroh - Avroh": {
                "description": "The ascending and descending patterns of notes that define the raga's melodic structure",
                "format": "Ascending pattern - Descending pattern (e.g., S R G M P D N S' - S' N D P M G R S)",
                "special_notation": "Separated by a hyphen (-), may include multiple paths (alternatives separated by 'or')"
            },
            "Pakad": {
                "description": "Characteristic phrases that uniquely identify the raga",
                "format": "Sequence of notes with phrase separators (;)",
                "importance": "These phrases are essential for recognizing and distinguishing ragas"
            },
            "Vadi-Samvadi": {
                "description": "The most important notes in the raga",
                "format": "Vadi - Samvadi (e.g., Madhyam - Shadj)",
                "vadi": "The dominant or most prominent note (sonant)",
                "samvadi": "The second most important note, usually at a fifth or fourth interval from Vadi (consonant)"
            },
            "Time": {
                "description": "Traditional time period when the raga is performed",
                "format": "Time range (e.g., 6AM to 9AM)",
                "importance": "Ragas are associated with specific times of day or seasons based on their mood and character"
            },
            "Swara - Set": {
                "description": "Complete set of notes (swaras) used in the raga",
                "format": "S-R-G-M-P-D-N-S (with appropriate variations marked as flat/sharp)",
                "notation": "Hyphen-separated list with lowercase letters for flat notes (komal) and uppercase for natural (shuddha)"
            },
            "Gamak": {
                "description": "Specific ornamentations and microtonal inflections characteristic to the raga",
                "format": "Sequence showing ornamentations with tilde () indicating oscillation (e.g., G)",
                "importance": "These ornamentations are crucial for authentic raga performance and recognition"
            },
            "Rasa": {
                "description": "The emotional essence or mood evoked by the raga",
                "common_values": [
                    "Shringar (Love/Beauty)",
                    "Hasya (Laughter/Mirth)",
                    "Karuna (Compassion/Pathos)",
                    "Raudra (Anger/Fury)",
                    "Veera (Heroism/Courage)",
                    "Bhayanak (Fear/Terror)",
                    "Vibhatsaya (Disgust/Aversion)",
                    "Adbhuta (Wonder/Amazement)",
                    "Shanta (Peace/Tranquility)"
                ],
                "importance": "The rasa helps musicians and listeners connect emotionally with the raga"
            }
        },
        "music_terms": {
            "Aaroh": "Ascending sequence of notes",
            "Avroh": "Descending sequence of notes",
            "Pakad": "Characteristic phrase that identifies the raga",
            "Vadi": "Most important note (dominant) in the raga",
            "Samvadi": "Second most important note (sub-dominant) in the raga",
            "Gamak": "Ornamentations and microtonal inflections specific to the raga",
            "Thaat": "Parent scale from which ragas are derived (there are 10 main thaats)",
            "Komal": "Flat note (indicated by lowercase letters)",
            "Shuddha": "Natural note (indicated by uppercase letters)",
            "Tivra": "Sharp note (primarily applies to Ma)",
            "Meend": "Gliding smoothly from one note to another",
            "Kampan": "Oscillation or vibrato between adjacent notes",
            "Murki": "Quick ornamentation involving multiple notes",
            "Khatka": "Sharp deflection or turn between notes"
        }
    }

# Detect gamakas (ornamentations) in the audio
def detect_gamakas(pitch_values, sr=22050, hop_length=512):
    """
    Detect gamakas (ornamentations) in the pitch contour.
    Returns gamaka features vector.
    """
    if len(pitch_values) == 0 or np.all(pitch_values <= 0):
        return np.zeros(5)  # Return zeros if no valid pitch
    
    # Filter out zero or negative values
    valid_pitch = pitch_values[pitch_values > 0]
    
    if len(valid_pitch) == 0:
        return np.zeros(5)
    
    # Smooth the pitch contour
    smoothed_pitch = gaussian_filter1d(valid_pitch, sigma=2)
    
    # Calculate the pitch derivative (velocity)
    pitch_derivative = np.diff(smoothed_pitch)
    
    # Calculate features that characterize gamakas
    # 1. Oscillation count (zero crossings)
    zero_crossings = np.sum(np.diff(np.signbit(pitch_derivative)))
    
    # 2. Pitch range (difference between max and min pitch)
    pitch_range = np.max(valid_pitch) - np.min(valid_pitch)
    
    # 3. Average pitch velocity
    avg_velocity = np.mean(np.abs(pitch_derivative))
    
    # 4. Percentage of time with significant movement
    significant_movement = np.sum(np.abs(pitch_derivative) > 5) / len(pitch_derivative)
    
    # 5. Pitch variance as measure of instability
    pitch_variance = np.var(valid_pitch)
    
    # Combine into feature vector
    gamaka_features = np.array([
        zero_crossings / len(valid_pitch),
        pitch_range,
        avg_velocity,
        significant_movement,
        pitch_variance
    ])
    
    return gamaka_features

# Detect Vadi (dominant) and Samvadi (sub-dominant) notes
def detect_vadi_samvadi(pitch_values, sr=22050):
    """
    Detect the vadi (dominant) and samvadi (sub-dominant) notes in the audio.
    """
    if len(pitch_values) == 0 or np.all(pitch_values <= 0):
        return ('S', 'P')  # Default if no valid pitch
    
    # Filter out zero or negative values
    valid_pitch = pitch_values[pitch_values > 0]
    
    if len(valid_pitch) == 0:
        return ('S', 'P')
    
    # Convert to Indian swaras
    swaras = improved_pitch_to_swara(valid_pitch, sr)
    
    # Count occurrence of each swara
    if not swaras:
        return ('S', 'P')
    
    swara_counts = Counter(swaras)
    
    # Get the most common (vadi) and second most common (samvadi)
    most_common = swara_counts.most_common(2)
    
    if len(most_common) >= 2:
        vadi, samvadi = most_common[0][0], most_common[1][0]
    elif len(most_common) == 1:
        vadi = most_common[0][0]
        samvadi = 'P' if vadi != 'P' else 'S'  # Default samvadi
    else:
        vadi, samvadi = 'S', 'P'  # Default
    
    return (vadi, samvadi)

# Segment audio into sections for analysis
def segment_audio(y_audio, sr=22050, segment_length=10, hop_size=5):
    """
    Segment audio into overlapping chunks for analysis.
    Returns list of segments with their start times.
    """
    segment_samples = segment_length * sr
    hop_samples = hop_size * sr
    
    # If audio is shorter than segment length, return the whole audio
    if len(y_audio) <= segment_samples:
        return [(y_audio, 0)]
    
    segments = []
    for start in range(0, len(y_audio) - segment_samples, hop_samples):
        segment = y_audio[start:start + segment_samples]
        segments.append((segment, start / sr))
    
    # Add the last segment if needed
    if start + hop_samples < len(y_audio) - segment_samples:
        start = len(y_audio) - segment_samples
        segments.append((y_audio[start:], start / sr))
    
    return segments

# Analyze audio segments to find the most representative raga section
def analyze_segments(y_audio, sr=22050):
    """
    Analyze audio segments to find sections that best represent each detected raga.
    """
    segments = segment_audio(y_audio, sr)
    segment_results = []
    
    for segment, start_time in segments:
        # Get pitch for this segment
        pitch = get_pitch(segment, sr)
        
        # Get notes
        swara_notes = improved_pitch_to_swara(pitch, sr)
        
        # Create pitch histogram
        pitch_histogram = create_pitch_histogram(pitch, sr)
        
        # Detect vadi and samvadi
        vadi_samvadi = detect_vadi_samvadi(pitch, sr)
        
        # Detect gamakas
        gamaka_features = detect_gamakas(pitch, sr)
        
        # Match raga
        raga_name, confidence, details = find_best_matching_raga(
            swara_notes, 
            pitch_histogram, 
            vadi_samvadi, 
            gamaka_features
        )
        
        segment_results.append({
            'start_time': start_time,
            'end_time': start_time + (len(segment) / sr),
            'raga': raga_name,
            'confidence': confidence,
            'vadi_samvadi': vadi_samvadi,
            'swaras': ''.join(swara_notes[:20]) + ('...' if len(swara_notes) > 20 else '')
        })
    
    return segment_results

# Extract features for CNN model and MATLAB-like analysis
def extract_features_from_file(file_path):
    # Load audio
    y_audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features for ML model if available
    if model_loaded:
        # Get raw pitch for rule-based matching
        pitch = get_pitch(y_audio, sr)
        
        # Extract 2D spectrogram features
        log_mel_spec = features_extractor(file_path)
    else:
        log_mel_spec = None
        pitch = None
    
    return log_mel_spec, pitch, y_audio, sr

# Function to perform comprehensive audio analysis (MATLAB-style)
def analyze_audio_signal(y_audio, sr):
    # Create time vector
    N = len(y_audio)
    t = np.arange(N) / sr
    
    # Basic signal statistics
    maxval = np.max(y_audio)
    minval = np.min(y_audio)
    mean_val = np.mean(y_audio)
    std_val = np.std(y_audio)
    
    # Dynamic range calculation
    non_zero_min = np.min(np.abs(y_audio[np.abs(y_audio) > 0]))
    dynamic_range = 20 * np.log10(maxval / non_zero_min)
    
    # Crest factor
    crest_factor = 20 * np.log10(maxval / std_val)
    
    # Calculate kurtosis and skewness
    kurt = kurtosis(y_audio)
    skewness = skew(y_audio)
    
    # Frequency domain analysis
    # FFT
    n_fft = 2048
    Y = np.fft.fft(y_audio, n_fft)
    Y_mag = np.abs(Y[:n_fft//2+1])
    f = np.linspace(0, sr/2, n_fft//2+1)
    
    # Power spectral density using Welch's method
    f_welch, pxx_welch = signal.welch(y_audio, sr, nperseg=1024)
    
    # Periodogram
    f_per, pxx_per = signal.periodogram(y_audio, sr)
    
    # Compute spectral centroid and bandwidth
    spectral_centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)[0]
    
    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y_audio)[0]
    
    # Compute autocorrelation
    autocorr = np.correlate(y_audio, y_audio, mode='full')
    autocorr = autocorr[N-1:] / autocorr[N-1]  # Normalize
    lag_time = np.arange(len(autocorr)) / sr
    
    # Find autocorrelation time (where autocorr drops below 0.05)
    try:
        ac_threshold = 0.05
        ac_time_index = np.where(autocorr < ac_threshold)[0][0]
        ac_time = lag_time[ac_time_index]
    except IndexError:
        ac_time = 0
    
    # Create graphs directory
    os.makedirs("static/graphs", exist_ok=True)
    graph_id = uuid.uuid4().hex
    
    # 1. Plot time domain waveform
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_audio, 'r')
    plt.xlim([0, max(t)])
    plt.ylim([-1.1*maxval, 1.1*maxval])
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal in Time Domain')
    waveform_file = f"waveform_{graph_id}.png"
    waveform_path = f"static/graphs/{waveform_file}"
    plt.savefig(waveform_path)
    plt.close()
    
    # 2. Plot spectrogram
    plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_file = f"spectrogram_{graph_id}.png"
    spectrogram_path = f"static/graphs/{spectrogram_file}"
    plt.savefig(spectrogram_path)
    plt.close()
    
    # 3. Plot periodogram (first version) - NEW
    plt.figure(figsize=(10, 6))
    pxx = signal.periodogram(y_audio, fs=sr)[1]
    plt.plot(10*np.log10(pxx))
    plt.grid(True)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Periodogram - Power Spectral Density')
    periodogram1_file = f"periodogram1_{graph_id}.png"
    periodogram1_path = f"static/graphs/{periodogram1_file}"
    plt.savefig(periodogram1_path)
    plt.close()
    
    # 4. Plot amplitude spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, 20*np.log10(Y_mag), 'r')
    plt.grid(True)
    plt.xlim([20, sr/2])  # Focus on audible range
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Amplitude Spectrum')
    spectrum_file = f"spectrum_{graph_id}.png"
    spectrum_path = f"static/graphs/{spectrum_file}"
    plt.savefig(spectrum_path)
    plt.close()
    
    # 5. Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_audio, bins=100)
    plt.grid(True)
    plt.xlabel('Signal Amplitude')
    plt.ylabel('Number of Samples')
    plt.title('Probability Distribution / Histogram')
    histogram_file = f"histogram_{graph_id}.png"
    histogram_path = f"static/graphs/{histogram_file}"
    plt.savefig(histogram_path)
    plt.close()
    
    # 6. Plot autocorrelation
    plt.figure(figsize=(10, 6))
    plt.plot(lag_time, autocorr, 'r')
    plt.grid(True)
    plt.xlim([0, min(0.5, max(lag_time))])  # Limit to 0.5s for better visualization
    plt.xlabel('Delay (s)')
    plt.ylabel('Autocorrelation Coefficient')
    plt.title('Autocorrelation of the Signal')
    plt.axhline(y=0.05, color='k', linestyle='--', alpha=0.7)
    autocorr_file = f"autocorr_{graph_id}.png"
    autocorr_path = f"static/graphs/{autocorr_file}"
    plt.savefig(autocorr_path)
    plt.close()
    
    # 7. Plot FFT of Speech Signal - NEW
    plt.figure(figsize=(10, 6))
    # Compute FFT and get only positive frequencies
    NFFT = 2**np.ceil(np.log2(len(y_audio))).astype(int)
    fft_freqs = np.linspace(0, sr/2, NFFT//2+1)
    fft_mag = np.abs(np.fft.fft(y_audio, NFFT))[:NFFT//2+1]
    plt.plot(fft_freqs, fft_mag)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Plot of Speech Signal')
    fft_file = f"fft_plot_{graph_id}.png"
    fft_path = f"static/graphs/{fft_file}"
    plt.savefig(fft_path)
    plt.close()
    
    # 8. Power spectral density using Welch's method
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_welch, pxx_welch)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.title('Power Spectral Density - Welch Method')
    psd_file = f"psd_{graph_id}.png"
    psd_path = f"static/graphs/{psd_file}"
    plt.savefig(psd_path)
    plt.close()
    
    # 9. Plot PSD using periodogram with normalized frequency - NEW
    plt.figure(figsize=(10, 6))
    # Compute normalized PSD
    N = len(y_audio)
    xdft = np.fft.fft(y_audio)
    xdft = xdft[:N//2+1]
    psdx = (1/(2*np.pi*N)) * np.abs(xdft)**2
    psdx[1:-1] = 2*psdx[1:-1]  # Double values for positive frequencies (except DC and Nyquist)
    freq = np.linspace(0, np.pi, N//2+1)  # Normalized frequency
    
    plt.plot(freq/np.pi, 10*np.log10(psdx))
    plt.grid(True)
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Power/Frequency (dB/rad/sample)')
    plt.title('Periodogram Using FFT - Normalized Frequency')
    norm_periodogram_file = f"norm_periodogram_{graph_id}.png"
    norm_periodogram_path = f"static/graphs/{norm_periodogram_file}"
    plt.savefig(norm_periodogram_path)
    plt.close()
    
    # 10. Plot pitch class histogram
    pitch = get_pitch(y_audio, sr)
    pitch_histogram = create_pitch_histogram(pitch, sr)
    
    plt.figure(figsize=(10, 6))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 
                  'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.bar(note_names[:24], pitch_histogram)
    plt.xlabel('Pitch Class')
    plt.ylabel('Normalized Frequency')
    plt.title('Pitch Class Histogram')
    pitch_hist_file = f"pitch_histogram_{graph_id}.png"
    pitch_hist_path = f"static/graphs/{pitch_hist_file}"
    plt.savefig(pitch_hist_path)
    plt.close()
    
    # 11. Plot pitch contour
    pitch_times = librosa.times_like(pitch, sr=sr)
    plt.figure(figsize=(10, 6))
    plt.plot(pitch_times, pitch)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Contour')
    pitch_contour_file = f"pitch_contour_{graph_id}.png"
    pitch_contour_path = f"static/graphs/{pitch_contour_file}"
    plt.savefig(pitch_contour_path)
    plt.close()
    
    # Return analysis results and graph paths
    analysis_results = {
        "time_domain": {
            "max_value": float(maxval),
            "min_value": float(minval),
            "mean_value": float(mean_val),
            "rms_value": float(std_val),  # In zero-mean signals, std is equivalent to RMS
            "dynamic_range_db": float(dynamic_range),
            "crest_factor_db": float(crest_factor),
            "kurtosis": float(kurt),
            "skewness": float(skewness)
        },
        "frequency_domain": {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "autocorrelation_time": float(ac_time)
        },
        "graphs": {
            "waveform": f"/graphs/{waveform_file}",
            "spectrogram": f"/graphs/{spectrogram_file}",
            "periodogram": f"/graphs/{periodogram1_file}",
            "spectrum": f"/graphs/{spectrum_file}",
            "histogram": f"/graphs/{histogram_file}",
            "autocorrelation": f"/graphs/{autocorr_file}",
            "fft_plot": f"/graphs/{fft_file}",
            "psd": f"/graphs/{psd_file}",
            "normalized_periodogram": f"/graphs/{norm_periodogram_file}",
            "pitch_histogram": f"/graphs/{pitch_hist_file}",
            "pitch_contour": f"/graphs/{pitch_contour_file}"
        }
    }
    
    return analysis_results

class PersonalizedRecommendationSystem:
    """
    Recommendation system that provides personalized raga and practice suggestions
    based on user's performance history and vocal characteristics
    """
    def __init__(self):
        self.model_path = "models/personalization/recommendation_model.pkl"
        self.scaler_path = "models/personalization/feature_scaler.pkl"
        
        # Try to load existing model, otherwise create a new one
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.model_trained = True
        except:
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.scaler = StandardScaler()
            self.model_trained = False
    
    def _extract_features(self, user_profile, performance_data):
        """Extract features from user profile and performance data"""
        features = []
        
        # Add vocal range features
        vocal_range = user_profile.get("vocal_range", {})
        features.append(vocal_range.get("min", 0) or 0)
        features.append(vocal_range.get("max", 0) or 0)
        
        # Add skill metrics
        personalization_data = user_profile.get("personalization_data", {})
        features.append(personalization_data.get("pitch_accuracy", 0) or 0)
        features.append(personalization_data.get("rhythm_stability", 0) or 0)
        features.append(personalization_data.get("gamaka_proficiency", 0) or 0)
        features.append(personalization_data.get("breath_control", 0) or 0)
        
        # Add performance metrics if available
        if performance_data:
            features.append(performance_data.get("overall_score", 50))
            features.append(performance_data.get("structure_adherence", {}).get("aaroh", 0))
            features.append(performance_data.get("structure_adherence", {}).get("pakad", 0))
            features.append(performance_data.get("vadi_samvadi_accuracy", 0))
            features.append(performance_data.get("rhythm_stability", 0))
        else:
            # Default values if no performance data
            features.extend([50, 0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def recommend_ragas(self, user_profile, performance_data=None, raga_df=None, count=3):
        """
        Recommend ragas based on user profile and performance data
        
        Args:
            user_profile (dict): User profile data
            performance_data (dict): Latest performance evaluation data
            raga_df (DataFrame): DataFrame containing raga information
            count (int): Number of ragas to recommend
            
        Returns:
            list: List of recommended ragas with reasons
        """
        if raga_df is None or len(raga_df) == 0:
            return []
        
        # Get user features
        features = self._extract_features(user_profile, performance_data)
        
        # If we don't have enough training data, use rule-based recommendations
        if not self.model_trained:
            return self._rule_based_recommendations(user_profile, performance_data, raga_df, count)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Get raga indices recommended by the model
        try:
            # Predict probabilities for all ragas
            raga_indices = np.argsort(self.model.predict_proba(scaled_features)[0])[-count:]
            
            recommendations = []
            for idx in reversed(raga_indices):
                raga_name = raga_df.iloc[idx]["name_of_the_raag"]
                
                # Generate a reason for recommendation
                reason = self._generate_recommendation_reason(user_profile, raga_df.iloc[idx])
                
                recommendations.append({
                    "raga": raga_name,
                    "reason": reason
                })
            
            return recommendations
        except:
            # Fall back to rule-based if model prediction fails
            return self._rule_based_recommendations(user_profile, performance_data, raga_df, count)
    
    def _rule_based_recommendations(self, user_profile, performance_data, raga_df, count=3):
        """Generate recommendations using rule-based approach"""
        recommendations = []
        
        # Get user skill level
        skill_level = user_profile.get("skill_level", "beginner")
        
        # Get previously practiced ragas
        practiced_ragas = set(user_profile.get("ragas_practiced", []))
        
        # Filter suitable ragas based on skill level
        if skill_level == "beginner":
            # For beginners, recommend ragas with simpler structures
            simple_ragas = []
            for _, row in raga_df.iterrows():
                raga_name = row["name_of_the_raag"]
                aaroh_avroh = row.get("aaroh_-_avroh", "")
                pakad = row.get("pakad", "")
                
                # Simple heuristic: shorter patterns are easier
                aaroh_avroh_tokens = len(tokenize_swaras(aaroh_avroh))
                pakad_tokens = len(tokenize_swaras(pakad))
                
                if aaroh_avroh_tokens <= 14 and pakad_tokens <= 10:
                    simple_ragas.append((raga_name, row))
            
            # Choose random ragas from simple ragas, preferring those not practiced yet
            candidates = [r for r in simple_ragas if r[0] not in practiced_ragas]
            if len(candidates) < count:
                # Add some practiced ragas if needed
                candidates.extend([r for r in simple_ragas if r[0] in practiced_ragas])
            
            # Select random recommendations
            for i in range(min(count, len(candidates))):
                raga_name, row = random.choice(candidates)
                candidates.remove((raga_name, row))
                
                reason = f"{raga_name} has a simpler structure suitable for beginners."
                
                recommendations.append({
                    "raga": raga_name,
                    "reason": reason
                })
                
        elif skill_level == "intermediate":
            # For intermediate users, recommend ragas with moderate complexity
            for _, row in raga_df.iterrows():
                raga_name = row["name_of_the_raag"]
                
                # Skip if already at the recommendation limit
                if len(recommendations) >= count:
                    break
                
                # Skip already practiced ragas if we have other options
                if raga_name in practiced_ragas and len(practiced_ragas) < len(raga_df) / 2:
                    continue
                
                # Simple heuristic for intermediate difficulty
                aaroh_avroh = row.get("aaroh_-_avroh", "")
                aaroh_avroh_tokens = len(tokenize_swaras(aaroh_avroh))
                
                if 15 <= aaroh_avroh_tokens <= 25:
                    reason = f"{raga_name} offers a good challenge appropriate to your intermediate level."
                    
                    recommendations.append({
                        "raga": raga_name,
                        "reason": reason
                    })
        else:
            # For advanced users, recommend complex ragas or those with specific features
            for _, row in raga_df.iterrows():
                raga_name = row["name_of_the_raag"]
                
                # Skip if already at the recommendation limit
                if len(recommendations) >= count:
                    break
                
                # For advanced users, look for complex gamakas or unique features
                gamaka = row.get("gamak", "")
                
                if gamaka and len(gamaka.strip()) > 0:
                    reason = f"{raga_name} features complex gamakas that will challenge your advanced skills."
                    
                    recommendations.append({
                        "raga": raga_name,
                        "reason": reason
                    })
        
        # If we still don't have enough recommendations, add random ones
        while len(recommendations) < count and len(recommendations) < len(raga_df):
            idx = random.randint(0, len(raga_df) - 1)
            raga_name = raga_df.iloc[idx]["name_of_the_raag"]
            
            # Skip if already recommended
            if any(rec["raga"] == raga_name for rec in recommendations):
                continue
            
            reason = f"{raga_name} would be an interesting next raga to explore."
            recommendations.append({
                "raga": raga_name,
                "reason": reason
            })
        
        return recommendations
    
    def _generate_recommendation_reason(self, user_profile, raga_info):
        """Generate a personalized reason for recommending a raga"""
        raga_name = raga_info["name_of_the_raag"]
        
        # Get user characteristics
        skill_level = user_profile.get("skill_level", "beginner")
        vocal_range = user_profile.get("vocal_range", {})
        min_pitch = vocal_range.get("min_hz", 0)
        max_pitch = vocal_range.get("max_hz", 0)
        
        personalization_data = user_profile.get("personalization_data", {})
        pitch_accuracy = personalization_data.get("pitch_accuracy", 0)
        rhythm_stability = personalization_data.get("rhythm_stability", 0)
        
        # Generate reason based on raga characteristics and user profile
        reasons = []
        
        # Check if raga matches user's vocal range
        if min_pitch and max_pitch:
            reasons.append(f"{raga_name} aligns well with your vocal range")
        
        # Check if user needs to work on particular skills
        if pitch_accuracy < 70:
            aaroh_avroh = raga_info.get("aaroh_-_avroh", "")
            if len(tokenize_swaras(aaroh_avroh)) < 20:
                reasons.append("its clear patterns will help improve your pitch accuracy")
        
        if rhythm_stability < 70:
            reasons.append("practicing this raga will help develop your rhythmic stability")
        
        # Add raga-specific reason
        time_of_day = raga_info.get("time_of_day", "")
        if time_of_day:
            reasons.append(f"it's traditionally performed during {time_of_day}")
        
        # Skill level specific reasons
        if skill_level == "beginner":
            pakad = raga_info.get("pakad", "")
            if pakad and len(tokenize_swaras(pakad)) < 10:
                reasons.append("it has a manageable pakad (characteristic phrase) for beginners")
        elif skill_level == "intermediate":
            reasons.append("it offers a good progression from your current skill level")
        else:
            gamak = raga_info.get("gamak", "")
            if gamak:
                reasons.append("its complex gamakas will challenge your advanced skills")
        
        # Combine reasons
        if reasons:
            return f"Recommended because {' and '.join(reasons)}."
        else:
            return f"{raga_name} would be an excellent next raga to explore based on your profile."
    
    def recommend_practice_routine(self, user_profile, performance_data=None):
        """
        Generate personalized practice routine based on user profile and performance data
        
        Args:
            user_profile (dict): User profile data
            performance_data (dict): Latest performance evaluation data
            
        Returns:
            dict: Personalized practice routine
        """
        # Get user skill level and personalization data
        skill_level = user_profile.get("skill_level", "beginner")
        personalization_data = user_profile.get("personalization_data", {})
        
        # Identify areas for improvement
        weak_areas = []
        for metric, value in personalization_data.items():
            if value < 70:  # Threshold for "needs improvement"
                weak_areas.append(metric)
        
        # Default practice routine
        routine = {
            "daily_practice_time": 20,  # minutes
            "exercises": []
        }
        
        # Adjust practice time based on skill level
        if skill_level == "intermediate":
            routine["daily_practice_time"] = 30
        elif skill_level == "advanced":
            routine["daily_practice_time"] = 45
        
        # Basic exercises that everyone should do
        routine["exercises"].append({
            "name": "Warm-up Sa-Re-Ga-Ma exercises",
            "duration": 5,
            "description": "Practice basic ascending and descending patterns to warm up your voice",
            "importance": "high"
        })
        
        # Add exercises based on weak areas
        exercise_time_remaining = routine["daily_practice_time"] - 5  # Subtract warm-up time
        
        if "pitch_accuracy" in weak_areas:
            minutes = min(10, exercise_time_remaining)
            exercise_time_remaining -= minutes
            
            routine["exercises"].append({
                "name": "Pitch matching exercise",
                "duration": minutes,
                "description": "Practice matching pitch with a tanpura. Focus on holding steady notes and transitioning smoothly between them.",
                "importance": "high"
            })
        
        if "rhythm_stability" in weak_areas:
            minutes = min(8, exercise_time_remaining)
            exercise_time_remaining -= minutes
            
            routine["exercises"].append({
                "name": "Rhythm exercises with tabla",
                "duration": minutes,
                "description": "Practice maintaining a steady rhythm with tabla accompaniment. Focus on the 'sam' (first beat) and consistent tempo.",
                "importance": "high"
            })
        
        if "gamaka_proficiency" in weak_areas:
            minutes = min(10, exercise_time_remaining)
            exercise_time_remaining -= minutes
            
            routine["exercises"].append({
                "name": "Gamaka practice",
                "duration": minutes,
                "description": "Practice essential ornamentations (meend, andolan, kan swar) on different notes.",
                "importance": "medium"
            })
        
        if "breath_control" in weak_areas:
            minutes = min(5, exercise_time_remaining)
            exercise_time_remaining -= minutes
            
            routine["exercises"].append({
                "name": "Breath control exercises",
                "duration": minutes,
                "description": "Practice long sustained notes while maintaining consistent volume and quality.",
                "importance": "medium"
            })
        
        # Additional exercises based on performance data
        if performance_data:
            structure_adherence = performance_data.get("structure_adherence", {})
            
            # Check for issues with pakad
            if structure_adherence.get("pakad", 100) < 70:
                minutes = min(8, exercise_time_remaining)
                exercise_time_remaining -= minutes
                
                routine["exercises"].append({
                    "name": "Pakad practice",
                    "duration": minutes,
                    "description": "Focus on the characteristic phrase (pakad) of your current raga. Repeat it until you can perform it accurately and confidently.",
                    "importance": "high"
                })
            
            # Check for issues with aaroh/avroh
            if (structure_adherence.get("aaroh", 100) + structure_adherence.get("avroh", 100)) / 2 < 70:
                minutes = min(8, exercise_time_remaining)
                exercise_time_remaining -= minutes
                
                routine["exercises"].append({
                    "name": "Aaroh-Avroh practice",
                    "duration": minutes,
                    "description": "Practice the ascending and descending patterns of your current raga, focusing on proper note transitions and phrasing.",
                    "importance": "high"
                })
        
        # Fill remaining time with general practice
        if exercise_time_remaining > 0:
            routine["exercises"].append({
                "name": "Free improvisation",
                "duration": exercise_time_remaining,
                "description": "Apply what you've learned in spontaneous melodic improvisation within your current raga's framework.",
                "importance": "medium"
            })
        
        # Add skill-level specific advice
        if skill_level == "beginner":
            routine["additional_advice"] = "Focus on pitch accuracy and clearly articulating each note."
        elif skill_level == "intermediate":
            routine["additional_advice"] = "Work on smooth transitions between notes and introducing basic gamakas."
        else:
            routine["additional_advice"] = "Develop your personal style while maintaining raga discipline."
        
        return routine
    
    def update_model(self, user_data, raga_df):
        """
        Update the recommendation model with new user data
        
        Args:
            user_data (dict): Dictionary mapping usernames to their profiles and performances
            raga_df (DataFrame): DataFrame containing raga information
            
        Returns:
            bool: True if model was updated successfully, False otherwise
        """
        try:
            # Prepare training data
            features = []
            targets = []
            
            for username, data in user_data.items():
                # Get user profile
                user_profile = data.get("profile", {})
                
                # Get performance data
                performances = data.get("performances", [])
                
                for performance in performances:
                    # Skip if no raga name
                    if "raga" not in performance:
                        continue
                    
                    # Find raga index
                    raga_name = performance["raga"]
                    raga_idx = -1
                    
                    for i, (_, row) in enumerate(raga_df.iterrows()):
                        if row["name_of_the_raag"] == raga_name:
                            raga_idx = i
                            break
                    
                    if raga_idx == -1:
                        continue  # Raga not found in DataFrame
                    
                    # Extract features
                    user_features = self._extract_features(user_profile, performance)
                    
                    features.append(user_features.flatten())
                    targets.append(raga_idx)
            
            # Train model if we have enough data
            if len(features) >= 3:
                X = np.array(features)
                y = np.array(targets)
                
                # Scale features
                self.scaler.fit(X)
                                # Scale features
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)

                # Train the model
                self.model.fit(X_scaled, y)

                # Save the updated model and scaler
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                self.model_trained = True
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error updating recommendation model: {e}")
            return False

# Initialize recommendation system
recommendation_system = PersonalizedRecommendationSystem()

# Flask Routes

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        user_dir = get_user_data_path(username)
        if os.path.exists(os.path.join(user_dir, 'profile.json')):
            return jsonify({'error': 'Username already exists'}), 400

        # Hash password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create user profile
        profile = {
            'username': username,
            'password': hashed_password,
            'created_at': datetime.now().isoformat(),
            'practice_sessions': 0,
            'total_practice_time': 0,
            'ragas_practiced': [],
            'skill_level': 'beginner',
            'vocal_range': {'min_hz': None, 'max_hz': None},
            'preferred_ragas': [],
            'achievements': [],
            'practice_streak': 0,
            'last_practice': None,
            'personalization_data': {
                'pitch_accuracy': 0.0,
                'rhythm_stability': 0.0,
                'gamaka_proficiency': 0.0,
                'breath_control': 0.0
            }
        }
        save_user_profile(username, profile)
        return jsonify({'message': 'Registration successful'}), 201
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        profile = load_user_profile(username)
        if not profile or not bcrypt.check_password_hash(profile['password'], password):
            return jsonify({'error': 'Invalid username or password'}), 401

        session['username'] = username
        return jsonify({'message': 'Login successful', 'username': username}), 200
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """User logout"""
    session.pop('username', None)
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/profile', methods=['GET', 'PUT'])
def profile():
    """Get or update user profile"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    username = session['username']
    profile = load_user_profile(username)

    if request.method == 'GET':
        # Remove password from response
        profile_copy = profile.copy()
        profile_copy.pop('password', None)
        return jsonify(profile_copy), 200

    elif request.method == 'PUT':
        try:
            data = request.get_json()
            # Update allowed fields
            for key in ['skill_level', 'vocal_range', 'preferred_ragas']:
                if key in data:
                    profile[key] = data[key]
            save_user_profile(username, profile)
            return jsonify({'message': 'Profile updated successfully'}), 200
        except Exception as e:
            logging.error(f"Profile update error: {e}")
            return jsonify({'error': 'Failed to update profile'}), 500

@app.route('/analyze_performance', methods=['POST'])
def analyze_performance():
    """Analyze uploaded audio for raga identification and performance evaluation"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    username = session['username']
    profile = load_user_profile(username)

    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith(('.wav', '.mp3')):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Save uploaded file temporarily
        temp_file = os.path.join('temp', f'{uuid.uuid4().hex}.wav')
        file.save(temp_file)

        # Extract features
        log_mel_spec, pitch, y_audio, sr = extract_features_from_file(temp_file)

        if log_mel_spec is None or pitch is None:
            return jsonify({'error': 'Failed to process audio'}), 500

        # CNN prediction
        if model_loaded:
            # Segment-based analysis
            segment_features = segment_features(temp_file)
            if segment_features is not None:
                cnn_preds = model.predict(segment_features)
                cnn_probs = np.mean(cnn_preds, axis=0)
                cnn_raga_idx = np.argmax(cnn_probs)
                cnn_raga = label_encoder.classes_[cnn_raga_idx]
                cnn_confidence = float(cnn_probs[cnn_raga_idx])
            else:
                cnn_raga = 'Unknown'
                cnn_confidence = 0.0
        else:
            cnn_raga = 'Unknown'
            cnn_confidence = 0.0

        # Rule-based matching
        swara_notes = improved_pitch_to_swara(pitch, sr)
        pitch_histogram = create_pitch_histogram(pitch, sr)
        vadi_samvadi = detect_vadi_samvadi(pitch, sr)
        gamaka_features = detect_gamakas(pitch, sr)
        rule_raga, rule_score, rule_details = find_best_matching_raga(
            swara_notes, pitch_histogram, vadi_samvadi, gamaka_features
        )

        # Hybrid prediction
        if model_loaded and cnn_raga != 'Unknown' and rule_raga != 'Unknown':
            # Convert rule scores to probabilities
            rule_probs = np.zeros(len(label_encoder.classes_))
            for raga_name, score in rule_details['top_matches']:
                if raga_name in label_encoder.classes_:
                    idx = np.where(label_encoder.classes_ == raga_name)[0][0]
                    rule_probs[idx] = score / 100
            final_raga_idx = hybrid_predict(cnn_probs, rule_probs)
            final_raga = label_encoder.classes_[final_raga_idx]
            final_confidence = float(cnn_probs[final_raga_idx] * 0.7 + rule_probs[final_raga_idx] * 0.3)
        else:
            final_raga = rule_raga if rule_raga != 'Unknown' else cnn_raga
            final_confidence = rule_score / 100 if rule_raga != 'Unknown' else cnn_confidence

        # Contextual filtering
        vocal_range = profile.get('vocal_range', {})
        filtered_ragas = filter_ragas_by_vocal_range(raga_df, vocal_range) if vocal_range.get('min_hz') else raga_df
        current_hour = datetime.now().hour
        time_filtered_ragas = filtered_ragas[filtered_ragas['time'].apply(
            lambda x: any(int(t.split(':')[0]) <= current_hour < int(t.split(':')[0]) + 3 for t in x.split('-'))
        )] if 'time' in filtered_ragas.columns else filtered_ragas

        # Check if predicted raga is in filtered list
        if final_raga not in time_filtered_ragas['name_of_the_raag'].values:
            final_raga = time_filtered_ragas.iloc[0]['name_of_the_raag'] if not time_filtered_ragas.empty else final_raga
            final_confidence *= 0.9  # Slight penalty for context mismatch

        # Get raga info for evaluation
        raga_info = raga_df[raga_df['name_of_the_raag'] == final_raga].iloc[0].to_dict() if final_raga in raga_df['name_of_the_raag'].values else {}

        # Transcribe audio for lyrics analysis
        transcribed_text = speech_to_text(temp_file)

        # Evaluate performance
        evaluation = evaluate_performance(y_audio, sr, raga_info, transcribed_text)

        # Analyze vocal characteristics
        vocal_analysis = analyze_vocal_characteristics(y_audio, sr)

        # Update user profile
        profile['practice_sessions'] += 1
        profile['total_practice_time'] += librosa.get_duration(y=y_audio, sr=sr) / 60
        profile['ragas_practiced'].append(final_raga)
        profile['vocal_range'] = {
            'min_hz': vocal_analysis['vocal_range']['min_hz'],
            'max_hz': vocal_analysis['vocal_range']['max_hz']
        }
        profile['personalization_data'].update({
            'pitch_accuracy': evaluation['overall_score'],
            'rhythm_stability': evaluation['rhythm_stability'],
            'gamaka_proficiency': np.mean(evaluation['detected_patterns']['gamaka_features']),
            'breath_control': vocal_analysis['breath_control']
        })
        profile['last_practice'] = datetime.now().isoformat()
        save_user_profile(username, profile)

        # Save performance record
        performance_data = {
            'raga': final_raga,
            'confidence': final_confidence,
            'evaluation': evaluation,
            'vocal_analysis': vocal_analysis,
            'transcribed_text': transcribed_text
        }
        save_performance_record(username, performance_data)

        # Generate recommendations
        recommendations = recommendation_system.recommend_ragas(profile, performance_data, raga_df)
        practice_routine = recommendation_system.recommend_practice_routine(profile, evaluation)

        # Signal analysis for visualization
        signal_analysis = analyze_audio_signal(y_audio, sr)

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return jsonify({
            'raga': final_raga,
            'confidence': final_confidence,
            'cnn_raga': cnn_raga,
            'cnn_confidence': cnn_confidence,
            'rule_raga': rule_raga,
            'rule_score': rule_score,
            'rule_details': rule_details,
            'evaluation': evaluation,
            'vocal_analysis': vocal_analysis,
            'transcribed_text': transcribed_text,
            'signal_analysis': signal_analysis,
            'recommendations': recommendations,
            'practice_routine': practice_routine
        }), 200

    except Exception as e:
        logging.error(f"Analysis error: {traceback.format_exc()}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/graphs/<filename>')
def serve_graph(filename):
    """Serve generated graph images"""
    return send_from_directory('static/graphs', filename)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get personalized recommendations"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    username = session['username']
    profile = load_user_profile(username)

    recommendations = recommendation_system.recommend_ragas(profile, raga_df)
    practice_routine = recommendation_system.recommend_practice_routine(profile)

    return jsonify({
        'recommendations': recommendations,
        'practice_routine': practice_routine
    }), 200

@app.route('/notation_guide', methods=['GET'])
def notation_guide():
    """Return Indian classical music notation guide"""
    return jsonify(get_notation_guide()), 200

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)