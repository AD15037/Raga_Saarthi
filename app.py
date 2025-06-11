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

# New imports for user management and ML personalization
from flask_bcrypt import Bcrypt
from flask_session import Session
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import shutil

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

def hz_to_swara(frequency):
    """
    Convert frequency in Hz to Indian classical music notation (swara)
    with appropriate octave information
    """
    if frequency <= 0:
        return "?"
    
    # Get the Western note first using librosa
    western_note = librosa.hz_to_note(frequency)
    
    # Parse the note and octave
    # Western note format is like 'C4' where C is the note and 4 is the octave
    note = western_note[:-1]  # Note part (like C, C#)
    octave = int(western_note[-1])  # Octave number
    
    # Convert to swara using the existing mapping
    if note in SWARA_MAPPING:
        swara = SWARA_MAPPING[note]
        
        # Add octave notation:
        # Middle octave (4) uses no symbol
        # Lower octave (3) uses a dot below, represented here as a comma prefix
        # Upper octave (5) uses a dot above, represented here as an apostrophe
        # For simplicity in display, using ',S' for lower octave and 'S'' for upper octave
        
        if octave < 4:
            # Lower octave
            return f",{swara}"
        elif octave > 4:
            # Upper octave
            return f"{swara}'"
        else:
            # Middle octave
            return swara
    else:
        return "?"

# Use this function in analyze_vocal_characteristics:
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
    
    # Get Western note names (internally needed for conversion)
    min_note_western = librosa.hz_to_note(min_pitch)
    max_note_western = librosa.hz_to_note(max_pitch)
    mean_note_western = librosa.hz_to_note(mean_pitch)
    
    # Convert to Indian classical swaras
    min_note_swara = hz_to_swara(min_pitch)
    max_note_swara = hz_to_swara(max_pitch)
    mean_note_swara = hz_to_swara(mean_pitch)
    
    # Calculate spectral characteristics for timbre
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0])
    
    # Get full spectral contrast for roughness calculation
    spectral_contrast_full = librosa.feature.spectral_contrast(y=audio_data, sr=sr)[0]
    
    # Brightness is related to spectral centroid
    brightness = spectral_centroid / (sr/2)  # Normalize by Nyquist frequency
    
    # Roughness can be estimated from spectral contrast
    if len(spectral_contrast_full) > 1:
        raw_roughness = np.mean(np.abs(np.diff(spectral_contrast_full)))
        # Normalize to 0-1 range with a reasonable maximum value
        roughness = min(1.0, raw_roughness / 5.0)  # Using 5.0 as a normalization factor
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
            "min_note": min_note_swara,    # Changed from min_note_western
            "max_note": max_note_swara,    # Changed from max_note_western
            "mean_note": mean_note_swara,  # Changed from mean_note_western
            # Keep these fields to maintain API compatibility, but use swara values
            "min_note_western": min_note_swara,  
            "max_note_western": max_note_swara,
            "mean_note_western": mean_note_swara,
            "min_swara": min_note_swara,
            "max_swara": max_note_swara,
            "mean_swara": mean_note_swara
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

def evaluate_performance(audio_data, sr, raga_info):
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
    
    # Calculate rhythm stability with enhanced analysis
    rhythm_metrics = analyze_rhythm_stability(audio_data, sr)
    rhythm_stability = rhythm_metrics['overall_rhythm_score']
    
    # Calculate overall performance score
    aaroh_avroh_score = (aaroh_adherence + avroh_adherence) / 2 if avroh else aaroh_adherence
    structure_score = 0.5 * aaroh_avroh_score + 0.5 * pakad_adherence if pakad else aaroh_avroh_score
    
    # Overall score combines structure adherence, vadi/samvadi accuracy and rhythm stability
    overall_score = 0.6 * structure_score + 0.4 * rhythm_stability
    
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
    if rhythm_metrics['overall_rhythm_score'] < 70:
        # Basic rhythm stability feedback
        feedback_message = "Work on maintaining consistent rhythm throughout your performance."
        
        # Add specific feedback based on detailed metrics
        if rhythm_metrics.get('tempo_stability', 100) < 60:
            feedback_message = "Focus on maintaining a steady tempo. Try practicing with a metronome."
        
        drift_direction = rhythm_metrics.get('drift_direction', '')
        drift_severity = rhythm_metrics.get('drift_severity', 0)
        
        if drift_severity > 30:
            if drift_direction == "rushing":
                feedback_message = "You tend to rush (speed up) during your performance. Focus on maintaining a steady tempo and practice with a metronome."
            elif drift_direction == "slowing":
                feedback_message = "You tend to slow down during your performance. Build stamina and concentration to maintain a consistent tempo throughout."
        
        feedback.append({
            "type": "rhythm",
            "area": "stability",
            "message": feedback_message,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in rhythm_metrics.items() if k != 'drift_direction'}
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

# Utility: extract swara tokens from a pattern string
def tokenize_swaras(swara_string):
    return [note.strip() for note in swara_string.replace('-', ' ').replace(',', ' ').split() if note.strip()]

# Utility: compare swara sequences
def match_score(input_seq, rule_seq):
    return SequenceMatcher(None, input_seq, rule_seq).ratio() * 100

# Improved: Match raga based on pakad, aaroh-avroh, and characteristic phrases
def find_best_matching_raga(pitch_notes, pitch_histogram=None, vadi_samvadi=None, gamaka_features=None):
    if not raga_rules_loaded:
        return "Unknown", 0, {}
    
    # Initialize result dictionary to store detailed matching scores
    matching_scores = {}
    best_raga = None
    best_score = 0
    best_details = {}

    for _, row in raga_df.iterrows():
        raga_name = row['name_of_the_raag']
        aaroh_avroh = row.get('aaroh_-_avroh', '')
        pakad = row.get('pakad', '')

        # Basic sequence matching
        rule_tokens = tokenize_swaras(f"{aaroh_avroh} {pakad}")
        sequence_score = match_score(pitch_notes, rule_tokens)
        
        # Initialize final score with the sequence matching score
        final_score = sequence_score
        score_details = {"sequence_match": sequence_score}
        
        # Add pitch histogram matching if available
        if pitch_histogram is not None and 'pitch_histogram' in row:
            try:
                raga_histogram = np.array([float(x) for x in row['pitch_histogram'].split(',')])
                histogram_score = np.corrcoef(pitch_histogram, raga_histogram)[0, 1] * 100
                final_score = 0.5 * final_score + 0.5 * histogram_score
                score_details["histogram_match"] = histogram_score
            except (ValueError, KeyError, IndexError):
                pass
        
        # Add vadi-samvadi matching if available
        if vadi_samvadi is not None and 'vadi' in row and 'samvadi' in row:
            try:
                vadi_match = 100 if vadi_samvadi[0] == row['vadi'] else 0
                samvadi_match = 100 if vadi_samvadi[1] == row['samvadi'] else 0
                vadi_score = 0.7 * vadi_match + 0.3 * samvadi_match
                final_score = 0.7 * final_score + 0.3 * vadi_score
                score_details["vadi_samvadi_match"] = vadi_score
            except (KeyError, IndexError):
                pass
        
        # Add gamaka features matching if available
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

    # Sort ragas by matching score for the top 3
    top_ragas = sorted(matching_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return best_raga, best_score, {"top_matches": top_ragas, "details": best_details}

# Feature extraction function - UPDATED to match the notebook implementation exactly
def features_extractor(file):
    try:
        audio, sample_rate = librosa.load(file, sr=None, duration=30)  # Limit to 30 seconds
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

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

# Create pitch histogram
def create_pitch_histogram(pitch_values, sr=22050, bins=12):
    """
    Create a normalized pitch class histogram from pitch values.
    """
    if len(pitch_values) == 0 or np.all(pitch_values <= 0):
        return np.zeros(bins)
    
    # Filter out zero or negative values
    valid_pitch = pitch_values[pitch_values > 0]
    
    if len(valid_pitch) == 0:
        return np.zeros(bins)
    
    # Convert frequencies to pitch classes (0-11, C to B)
    C4_FREQ = 261.63
    cents = 1200 * np.log2(valid_pitch / C4_FREQ)
    pitch_classes = (cents / 100) % 12
    
    # Create histogram
    histogram, _ = np.histogram(pitch_classes, bins=bins, range=(0, 12), density=True)
    
    # Normalize
    histogram = histogram / np.sum(histogram)
    
    return histogram

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
        pitch = librosa.yin(segment, fmin=librosa.note_to_hz('C1'), 
                           fmax=librosa.note_to_hz('C8'), 
                           sr=sr)
        
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
    print("Segment analysis results:", segment_results)  # Debugging line
    return segment_results

# Extract features for CNN model and MATLAB-like analysis - UPDATED to match notebook
def extract_features_from_file(file_path):
    # Load audio - using default parameters to match notebook
    y_audio, sr = librosa.load(file_path, sr=None)
    
    # Extract features for ML model if available
    if model_loaded:
        # Get raw pitch for rule-based matching
        pitch = librosa.yin(y_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        
        # Use EXACTLY the same feature extraction as in notebook
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
    
    # 7. FFT of Speech Signal - NEW
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
    
    # NEW: 10. Plot pitch class histogram
    pitch = librosa.yin(y_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), sr=sr)
    pitch_histogram = create_pitch_histogram(pitch, sr)
    
    plt.figure(figsize=(10, 6))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.bar(note_names, pitch_histogram)
    plt.xlabel('Pitch Class')
    plt.ylabel('Normalized Frequency')
    plt.title('Pitch Class Histogram')
    pitch_hist_file = f"pitch_histogram_{graph_id}.png"
    pitch_hist_path = f"static/graphs/{pitch_hist_file}"
    plt.savefig(pitch_hist_path)
    plt.close()
    
    # NEW: 11. Plot pitch contour
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
            # Add the /static prefix to all graph paths
            "waveform": f"/static/graphs/{waveform_file}",
            "spectrogram": f"/static/graphs/{spectrogram_file}",
            "periodogram": f"/static/graphs/{periodogram1_file}",
            "spectrum": f"/static/graphs/{spectrum_file}",
            "histogram": f"/static/graphs/{histogram_file}",
            "autocorrelation": f"/static/graphs/{autocorr_file}",
            "fft_plot": f"/static/graphs/{fft_file}",
            "psd": f"/static/graphs/{psd_file}",
            "normalized_periodogram": f"/static/graphs/{norm_periodogram_file}",
            "pitch_histogram": f"/static/graphs/{pitch_hist_file}",
            "pitch_contour": f"/static/graphs/{pitch_contour_file}"
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
        exercise_time_remaining = routine["daily_practice_time"] - 5 # Subtract warm-up time
        
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
            # Ensure at least 2 minutes for breath control
            minutes = max(2, min(5, exercise_time_remaining))
            exercise_time_remaining = max(0, exercise_time_remaining - minutes)
            
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
                # Ensure at least 3 minutes for pakad practice
                minutes = max(3, min(8, exercise_time_remaining))
                exercise_time_remaining = max(0, exercise_time_remaining - minutes)
                
                routine["exercises"].append({
                    "name": "Pakad practice",
                    "duration": minutes,
                    "description": "Focus on the characteristic phrase (pakad) of your current raga. Repeat it until you can perform it accurately and confidently.",
                    "importance": "high"
                })
            
            # Check for issues with aaroh/avroh
            if (structure_adherence.get("aaroh", 100) + structure_adherence.get("avroh", 100)) / 2 < 70:
                # Ensure at least 3 minutes for aaroh-avroh practice
                minutes = max(3, min(8, exercise_time_remaining))
                exercise_time_remaining = max(0, exercise_time_remaining - minutes)
                
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
                X_scaled = self.scaler.transform(X)
                
                # Train model
                self.model.fit(X_scaled, y)
                
                # Save model
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                
                self.model_trained = True
                return True
        except Exception as e:
            print(f"Error updating recommendation model: {e}")
        
        return False
        
# Create recommendation system instance
recommendation_system = PersonalizedRecommendationSystem()

def update_user_progress(username, performance_data, vocal_characteristics):
    """
    Update user progress and skill metrics based on performance data
    
    Args:
        username (str): Username
        performance_data (dict): Performance evaluation data
        vocal_characteristics (dict): Vocal characteristics data
        
    Returns:
        dict: Updated user profile
    """
    # Load user profile
    profile = load_user_profile(username)
    
    # Update practice statistics
    profile["practice_sessions"] += 1
    
    # Estimate session duration in minutes
    session_duration = 5  # Default 5 minutes
    profile["total_practice_time"] += session_duration
    
    # Update last practice time
    profile["last_practice"] = datetime.now().isoformat()
    
    # Update practice streak
    if profile["last_practice"]:
        last_practice = datetime.fromisoformat(profile["last_practice"])
        days_since_last = (datetime.now() - last_practice).days
        
        if days_since_last <= 1:  # Same day or consecutive day
            profile["practice_streak"] += 1
        else:
            profile["practice_streak"] = 1
    else:
        profile["practice_streak"] = 1
    
    # Update vocal range
    vocal_range = vocal_characteristics.get("vocal_range", {})
    
    if vocal_range:
        min_hz = vocal_range.get("min_hz", None)
        max_hz = vocal_range.get("max_hz", None)
        
        if min_hz and max_hz:
            # Update min vocal range
            if profile["vocal_range"]["min"] is None or min_hz < profile["vocal_range"]["min"]:
                profile["vocal_range"]["min"] = min_hz
            
            # Update max vocal range
            if profile["vocal_range"]["max"] is None or max_hz > profile["vocal_range"]["max"]:
                profile["vocal_range"]["max"] = max_hz
    
    # Update ragas practiced - use predicted_raga if available
    raga = performance_data.get("predicted_raga", performance_data.get("raga", ""))
    if raga and raga not in profile["ragas_practiced"]:
        profile["ragas_practiced"].append(raga)
    
    # Update skill metrics
    personalization_data = profile.get("personalization_data", {})
    
    # Update pitch accuracy
    new_pitch_accuracy = (performance_data.get("structure_adherence", {}).get("aaroh", 0) + 
                         performance_data.get("structure_adherence", {}).get("avroh", 0)) / 2
    
    # Update rhythm stability
    new_rhythm_stability = performance_data.get("rhythm_stability", 0)
    
    # Update gamaka proficiency based on gamaka features
    gamaka_features = performance_data.get("detected_patterns", {}).get("gamaka_features", [])
    new_gamaka_proficiency = 50.0  # Default
    if gamaka_features and len(gamaka_features) >= 3:
        # Simple heuristic for gamaka proficiency
        new_gamaka_proficiency = min(100, gamaka_features[0] * 10 + gamaka_features[2] * 20)
    
    # Update breath control from vocal characteristics
    new_breath_control = vocal_characteristics.get("breath_control", 0) * 100
    
    # Update metrics with exponential moving average (give more weight to recent performances)
    alpha = 0.3  # Smoothing factor
    
    personalization_data["pitch_accuracy"] = alpha * new_pitch_accuracy + (1 - alpha) * personalization_data.get("pitch_accuracy", new_pitch_accuracy)
    personalization_data["rhythm_stability"] = alpha * new_rhythm_stability + (1 - alpha) * personalization_data.get("rhythm_stability", new_rhythm_stability)
    personalization_data["gamaka_proficiency"] = alpha * new_gamaka_proficiency + (1 - alpha) * personalization_data.get("gamaka_proficiency", new_gamaka_proficiency)
    personalization_data["breath_control"] = alpha * new_breath_control + (1 - alpha) * personalization_data.get("breath_control", new_breath_control)
    
    # Generate video recommendations based on updated metrics - use predicted_raga if available
    raga = performance_data.get("predicted_raga", performance_data.get("raga", "Indian Classical"))
    video_recommendations = get_video_recommendations(
        personalization_data["pitch_accuracy"],
        personalization_data["rhythm_stability"],
        personalization_data["gamaka_proficiency"],
        personalization_data["breath_control"],
        raga
    )
    
    # Save video recommendations to profile for retrieval in the UI
    profile["recent_video_recommendations"] = video_recommendations
    
    # Update profile
    profile["personalization_data"] = personalization_data
    
    # Update skill level based on metrics
    avg_skill = (personalization_data["pitch_accuracy"] + 
                personalization_data["rhythm_stability"] + 
                personalization_data["gamaka_proficiency"] +
                personalization_data["breath_control"]) / 4
    
    if avg_skill > 80:
        profile["skill_level"] = "advanced"
    elif avg_skill > 60:
        profile["skill_level"] = "intermediate"
    else:
        profile["skill_level"] = "beginner"
    
    # Check for achievements
    if profile["practice_streak"] >= 7 and "7_day_streak" not in profile["achievements"]:
        profile["achievements"].append("7_day_streak")
    
    if len(profile["ragas_practiced"]) >= 5 and "5_ragas_learned" not in profile["achievements"]:
        profile["achievements"].append("5_ragas_learned")
    
    if profile["total_practice_time"] >= 60 and "1_hour_milestone" not in profile["achievements"]:
        profile["achievements"].append("1_hour_milestone")
    
    # Save updated profile
   
      
    save_user_profile(username, profile)
    
    return profile

# User management routes
@app.route("/register", methods=["POST"])
def register_user():
    """Register a new user"""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    user_dir = os.path.join(USER_DATA_PATH, username)
    if os.path.exists(user_dir):
        return jsonify({"error": "Username already exists"}), 400
    
    # Create user directory
    os.makedirs(user_dir, exist_ok=True)
    
    # Hash the password
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Create user profile
    profile = {
        "username": username,
                                                         "password_hash": password_hash,
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
    
    # Save profile
    save_user_profile(username, profile)
    
    # Log user in
    session["username"] = username
    
    return jsonify({"message": "User registered successfully"})

@app.route("/login", methods=["POST"])
def login_user():
    """Log in a user"""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    # Check if user exists
    profile_path = os.path.join(get_user_data_path(username), "profile.json")
    if not os.path.exists(profile_path):
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Load profile
    with open(profile_path, "r") as f:
        profile = json.load(f)
    
    # Check password
    password_hash = profile.get("password_hash")
    if not password_hash or not bcrypt.check_password_hash(password_hash, password):
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Log user in
    session["username"] = username
    
    # Return basic profile info (excluding password hash)
    profile_info = {k: v for k, v in profile.items() if k != "password_hash"}
    return jsonify({"message": "Login successful", "profile": profile_info})

@app.route("/logout", methods=["POST"])
def logout_user():
    """Log out a user"""
    session.pop("username", None)
    return jsonify({"message": "Logout successful"})

@app.route("/profile", methods=["GET"])
def get_user_profile():
    """Get the current user's profile"""
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    profile = load_user_profile(username)
    
    # Remove sensitive information
    if "password_hash" in profile:
        del profile["password_hash"]
    
    return jsonify(profile)

@app.route("/profile", methods=["PUT"])
def update_user_profile():
    """Update the current user's profile"""
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    profile = load_user_profile(username)
    
    # Update allowed fields
    allowed_fields = [
        "preferred_ragas"
    ]
    
    for field in allowed_fields:
        if field in data:
            profile[field] = data[field]
    
    # Save profile
    save_user_profile(username, profile)
    
    return jsonify({"message": "Profile updated successfully", "profile": profile})

# Performance analysis with personalization
@app.route("/analyze_performance", methods=["POST"])
def analyze_performance():
    """
    Analyze performance with personalized feedback based on user profile
    Requires user login
    """
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    raga_name = request.form.get('raga', '')
    
    if not raga_name:
        return jsonify({"error": "Please specify a raga name"}), 400
    
    # Find the raga info
    raga_info = None
    for _, row in raga_df.iterrows():
        if row['name_of_the_raag'].lower() == raga_name.lower():
            raga_info = row.to_dict()
            break
    
    if raga_info is None:
        return jsonify({"error": f"Raga '{raga_name}' not found in database"}), 404
        
    # Save temporary file
    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}.wav")
    file.save(temp_path)
    
    try:
        # Extract features for CNN model prediction
        features, pitch, y_audio, sr = extract_features_from_file(temp_path)
        
        # Use CNN model to predict raga if available
        cnn_prediction = None
        if model_loaded and features is not None:
            # Reshape features for model input
            features_reshaped = np.array(features).reshape(1, -1)
            
            # Predict raga using CNN
            prediction_probabilities = model.predict(features_reshaped)[0]
            top_indices = np.argsort(prediction_probabilities)[-3:][::-1]  # Get top 3 predictions
            
            # Get raga names and probabilities
            cnn_prediction = {
                "predicted_ragas": [
                    {
                        "name": label_encoder.inverse_transform([idx])[0],
                        "confidence": float(prediction_probabilities[idx] * 100)
                    } for idx in top_indices
                ],
                "selected_raga": raga_name,
                "match_confidence": 0.0  # Will update below if the selected raga is in predictions
            }
            
            # Calculate match confidence between user-selected raga and CNN prediction
            for pred in cnn_prediction["predicted_ragas"]:
                if pred["name"].lower() == raga_name.lower():
                    cnn_prediction["match_confidence"] = pred["confidence"]
                    break
        
        # Analyze vocal characteristics
        vocal_characteristics = analyze_vocal_characteristics(y_audio, sr)
        
        # Evaluate performance against user-selected raga
        selected_raga_results = evaluate_performance(y_audio, sr, raga_info)
        
        # Initialize performance results with selected raga evaluation
        performance_results = selected_raga_results.copy()
        
        # Add raga and timestamp to performance data
        performance_results["raga"] = raga_name
        performance_results["timestamp"] = datetime.now().isoformat()
        
        # Add CNN prediction to performance results
        predicted_raga_info = None
        if cnn_prediction:
            performance_results["cnn_prediction"] = cnn_prediction
            
            # Update the raga name to use the top predicted raga if confidence is high enough
            top_predicted_raga = cnn_prediction["predicted_ragas"][0]
            if top_predicted_raga["confidence"] > 75:  # 75% threshold for high confidence
                predicted_raga_name = top_predicted_raga["name"]
                performance_results["predicted_raga"] = predicted_raga_name
                
                # Find the predicted raga info
                for _, row in raga_df.iterrows():
                    if row['name_of_the_raag'].lower() == predicted_raga_name.lower():
                        predicted_raga_info = row.to_dict()
                        break
                
                # Re-evaluate performance against the predicted raga
                if predicted_raga_info:
                    predicted_raga_results = evaluate_performance(y_audio, sr, predicted_raga_info)
                    
                    # Replace scores with the predicted raga evaluation
                    performance_results["structure_adherence"] = predicted_raga_results["structure_adherence"]
                    performance_results["vadi_samvadi_accuracy"] = predicted_raga_results["vadi_samvadi_accuracy"]
                    performance_results["overall_score"] = predicted_raga_results["overall_score"]
                    performance_results["feedback"] = predicted_raga_results["feedback"]
                    
                    # Add note about score calculation
                    performance_results["score_basis"] = {
                        "based_on": "predicted_raga",
                        "predicted_raga": predicted_raga_name,
                        "selected_raga": raga_name
                    }
            else:
                performance_results["predicted_raga"] = raga_name
                performance_results["score_basis"] = {
                    "based_on": "selected_raga",
                    "selected_raga": raga_name
                }
        else:
            performance_results["predicted_raga"] = raga_name
            performance_results["score_basis"] = {
                "based_on": "selected_raga",
                "selected_raga": raga_name
            }
        
        # Save performance to user history
        save_performance_record(username, performance_results)
        
        # Update user progress
        updated_profile = update_user_progress(username, performance_results, vocal_characteristics)
        
        # Prepare response
        response_data = {
            "performance": performance_results,
            "vocal_characteristics": vocal_characteristics,
            "profile_updates": {
                "skill_level": updated_profile["skill_level"],
                "practice_streak": updated_profile["practice_streak"],
                "achievements": updated_profile["achievements"]
            },
            # Add video recommendations to response
            "video_recommendations": updated_profile.get("recent_video_recommendations", {})
        }
        
        # Create audio analysis visualization
        analysis_results = analyze_audio_signal(y_audio, sr)
        response_data["analysis"] = analysis_results
        
        # Add detailed rhythm analysis
        rhythm_metrics = analyze_rhythm_stability(y_audio, sr)
        
        # Generate rhythm visualization
        rhythm_viz_path = None  # Visualization not implemented
        
        # Generate rhythm exercises
        rhythm_exercises = generate_rhythm_exercises(rhythm_metrics)
        
        # Add to response
        response_data["rhythm_analysis"] = {
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in rhythm_metrics.items()},
            "visualization": rhythm_viz_path,
            "exercises": rhythm_exercises
        }
        
        os.remove(temp_path)
        return jsonify(response_data)
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Personalized recommendations
@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    """
    Get personalized raga and practice recommendations
    Requires user login
    """
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    # Load user profile
    profile = load_user_profile(username)
    
    # Check if the user has any performance data
    performances_path = os.path.join(get_user_data_path(username), "performances.json")
    latest_performance = None
    
    if os.path.exists(performances_path):
        with open(performances_path, "r") as f:
            performances = json.load(f)
            
        if performances:
            # Get the latest performance
            latest_performance = max(performances, key=lambda p: p.get("timestamp", ""))
    
    # Get raga recommendations
    raga_recommendations = recommendation_system.recommend_ragas(
        profile, latest_performance, raga_df, count=3
    )
    
    # Get practice routine
    practice_routine = recommendation_system.recommend_practice_routine(
        profile, latest_performance
    )
    
    return jsonify({
        "raga_recommendations": raga_recommendations,
        "practice_routine": practice_routine
    })

# Update recommendation model (admin function)
@app.route("/admin/update_recommendation_model", methods=["POST"])
def update_recommendation_model():
    """
    Update the recommendation model with all user data
    Admin access only (simplified for now)
    """
    # Simple admin check - would use proper authentication in production
    admin_key = request.headers.get("X-Admin-Key")
    if not admin_key or admin_key != "admin_secret_key":
        return jsonify({"error": "Unauthorized"}), 401
    
    # Collect all user data
    user_data = {}
    
    for username in os.listdir(USER_DATA_PATH):
        user_dir = os.path.join(USER_DATA_PATH, username)
        if not os.path.isdir(user_dir):
            continue
        
        # Load user profile
        profile_path = os.path.join(user_dir, "profile.json")
        if not os.path.exists(profile_path):
            continue
            
        with open(profile_path, "r") as f:
            profile = json.load(f)
        
        # Load performances
        performances_path = os.path.join(user_dir, "performances.json")
        performances = []
        
        if os.path.exists(performances_path):
            with open(performances_path, "r") as f:
                performances = json.load(f)
        
        # Add to user data
        user_data[username] = {
            "profile": profile,
            "performances": performances
        }
    print("User data collected for model update:", user_data)  # Debugging line
    
    # Update model
    success = recommendation_system.update_model(user_data, raga_df)
    
    if success:
        return jsonify({"message": "Recommendation model updated successfully"})
    else:
        return jsonify({"message": "Not enough data to update model"}), 400

# Progress tracking
@app.route("/progress", methods=["GET"])
def get_user_progress():
    """
    Get user progress metrics and history
    Requires user login
    """
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    # Load user profile
    profile = load_user_profile(username)
    
    # Load performances
    performances_path = os.path.join(get_user_data_path(username), "performances.json")
    performances = []
    
    if os.path.exists(performances_path):
        with open(performances_path, "r") as f:
            performances = json.load(f)
    
    # Calculate progress metrics
    metrics = {
        "sessions_completed": profile["practice_sessions"],
        "total_practice_time": profile["total_practice_time"],
        "current_streak": profile["practice_streak"],
        "ragas_learned": len(profile["ragas_practiced"]),
        "skill_metrics": profile["personalization_data"],
        "skill_level": profile["skill_level"],
        "achievements": profile["achievements"]
    }
    
    # Calculate improvement over time
    if len(performances) >= 2:
        first_perf = performances[0]
        last_perf = performances[-1]
        
        improvement = {
            "overall_score": last_perf.get("overall_score", 0) - first_perf.get("overall_score", 0),
            "days_practicing": (datetime.fromisoformat(last_perf.get("timestamp", datetime.now().isoformat())) - 
                             datetime.fromisoformat(first_perf.get("timestamp", datetime.now().isoformat()))).days
        }
        
        metrics["improvement"] = improvement
    
    # Get performance history for charts
    history = []
    for perf in performances:
        history.append({
            "timestamp": perf.get("timestamp"),
            "raga": perf.get("raga", ""),
            "overall_score": perf.get("overall_score", 0),
            "aaroh_adherence": perf.get("structure_adherence", {}).get("aaroh", 0),
            "avroh_adherence": perf.get("structure_adherence", {}).get("avroh", 0),
            "pakad_adherence": perf.get("structure_adherence", {}).get("pakad", 0),
            "rhythm_stability": perf.get("rhythm_stability", 0)
        })
    print("Progress metrics calculated:", metrics)  # Debugging line
    
    return jsonify({
        "metrics": metrics,
        "history": history
    })

# Enhanced rhythm stability analysis
@app.route("/analyze_rhythm", methods=["POST"])
def analyze_rhythm():
    """
    Analyze the rhythm stability of a performance
    """
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    
    # Save temporary file
    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}.wav")
    file.save(temp_path)
    
    try:
        # Load audio
        y_audio, sr = librosa.load(temp_path, sr=None)
        
        # Analyze rhythm stability
        metrics = analyze_rhythm_stability(y_audio, sr)
        
        os.remove(temp_path)
        return jsonify(metrics)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def analyze_rhythm_stability(audio_data, sr):
    """
    Perform comprehensive rhythm stability analysis
    Returns multiple metrics for rhythm quality assessment
    """
    metrics = {}
    
    # 1. Basic onset-based stability (existing approach, enhanced)
    try:
        # Detect onsets with improved parameters
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, 
                                           backtrack=True, units='time')
        
        if len(onsets) > 1:
            # Calculate inter-onset intervals
            iois = np.diff(onsets)
            
            # Basic stability metric (normalized IOI variance)
            metrics['basic_stability'] = 100.0 - min(100.0, (np.std(iois) / np.mean(iois)) * 100.0)
            
            # IOI consistency groups - detect how many distinct IOI clusters exist
            # (fewer clusters = more consistent rhythm)
            if len(iois) > 3:
                from sklearn.cluster import KMeans
                num_clusters = min(3, len(iois) - 1)
                kmeans = KMeans(n_clusters=num_clusters).fit(iois.reshape(-1, 1))
                cluster_sizes = np.bincount(kmeans.labels_)
                dominant_rhythm_ratio = np.max(cluster_sizes) / len(iois)
                metrics['rhythm_consistency'] = dominant_rhythm_ratio * 100
            else:
                metrics['rhythm_consistency'] = metrics['basic_stability']
    except Exception as e:
        print(f"Error in onset detection: {e}")
        metrics['basic_stability'] = 0.0
        metrics['rhythm_consistency'] = 0.0
    
    # 2. Tempo analysis
    try:
        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        if len(beat_frames) > 1:
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Calculate beat regularity
            beat_intervals = np.diff(beat_times)
            tempo_stability = 100.0 - min(100.0, (np.std(beat_intervals) / np.mean(beat_intervals)) * 100.0)
            metrics['tempo_stability'] = tempo_stability
            metrics['detected_tempo'] = float(tempo)
            
            # Calculate tempo drift over time
            if len(beat_intervals) > 4:
                # Use linear regression to check if tempo is speeding up or slowing down
                from scipy.stats import linregress
                x = np.arange(len(beat_intervals))
                slope, _, r_value, _, _ = linregress(x, beat_intervals)
                
                # Normalized slope (-1 to 1 range, where 0 is no drift)
                norm_slope = np.tanh(slope * 10)
                drift_direction = "steady"
                if norm_slope > 0.1:
                    drift_direction = "slowing"
                elif norm_slope < -0.1:
                    drift_direction = "rushing"
                
                metrics['tempo_drift'] = norm_slope
                metrics['drift_direction'] = drift_direction
                metrics['drift_severity'] = abs(norm_slope) * 100
    except Exception as e:
        print(f"Error in tempo analysis: {e}")
        metrics['tempo_stability'] = 0.0
        metrics['detected_tempo'] = 0.0
        metrics['tempo_drift'] = 0.0
        metrics['drift_direction'] = "unknown"
        metrics['drift_severity'] = 0.0
    
    # 3. Calculate overall rhythm score from component metrics
    component_scores = [
        metrics.get('basic_stability', 0),
        metrics.get('rhythm_consistency', 0),
        metrics.get('tempo_stability', 0),
        100 - metrics.get('drift_severity', 0)
    ]
    metrics['overall_rhythm_score'] = sum(score for score in component_scores if score > 0) / max(1, sum(1 for score in component_scores if score > 0))
    
    return metrics

def generate_rhythm_exercises(rhythm_metrics):
    """Generate tailored rhythm exercises based on detected rhythm issues"""
    exercises = []
    
    overall_score = rhythm_metrics.get('overall_rhythm_score', 0)
    drift_direction = rhythm_metrics.get('drift_direction', '')
    tempo_stability = rhythm_metrics.get('tempo_stability', 0)
    
    # Basic metronome practice
    exercises.append({
        "name": "Basic Metronome Practice",
        "duration": 10,
        "description": "Practice singing Sa with a metronome at 60 BPM. Gradually increase tempo as you improve.",
        "importance": "high"
    })
    
    # Add specific exercises based on issues
    if drift_direction == "rushing":
        exercises.append({
            "name": "Slow-motion Practice",
            "duration": 8,
            "description": "Practice at half your normal tempo, focusing on maintaining steadiness. Record yourself to check if you still rush.",
            "importance": "high"
        })
    
    elif drift_direction == "slowing":
        exercises.append({
            "name": "Stamina Building",
            "duration": 5,
            "description": "Practice holding notes for longer durations to build vocal stamina, which helps prevent slowing down.",
            "importance": "medium"
        })
    
    if tempo_stability < 60:
        exercises.append({
            "name": "Subdivision Practice",
            "duration": 8,
            "description": "Practice counting subdivisions (1-&-2-&-3-&-4-&) while performing to improve internal timing.",
            "importance": "high"
        })
    
    # Add tabla pattern practice for everyone
    exercises.append({
        "name": "Tabla Pattern Recognition",
        "duration": 5,
        "description": "Listen to and try to replicate common tabla patterns. Start with basic Teentaal patterns.",
        "importance": "medium"
    })
    
    return exercises

# YouTube video recommendations
@app.route("/video_recommendations", methods=["POST"])
def video_recommendations():
    """
    Get personalized YouTube video recommendations for skill improvement
    Requires user login
    """
    username = session.get("username")
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json()
    raga_name = data.get("raga_name", "")
    
    # Load user profile
    profile = load_user_profile(username)
    
    # Get latest performance data
    performances_path = os.path.join(get_user_data_path(username), "performances.json")
    latest_performance = None
    
    if os.path.exists(performances_path):
        with open(performances_path, "r") as f:
            performances = json.load(f)
            
        if performances:
            # Get the latest performance
            latest_performance = max(performances, key=lambda p: p.get("timestamp", ""))
    
    # Extract skill metrics
    pitch_accuracy = profile.get("personalization_data", {}).get("pitch_accuracy", 0)
    rhythm_stability = profile.get("personalization_data", {}).get("rhythm_stability", 0)
    gamaka_proficiency = profile.get("personalization_data", {}).get("gamaka_proficiency", 0)
    breath_control = profile.get("personalization_data", {}).get("breath_control", 0)
    
    # Generate recommendations
    recommendations = get_video_recommendations(pitch_accuracy, rhythm_stability, gamaka_proficiency, breath_control, raga_name)
    
    return jsonify(recommendations)

def get_video_recommendations(pitch_accuracy, rhythm_stability, gamaka_proficiency, breath_control, raga_name):
    """
    Generate personalized YouTube video recommendations based on user's performance metrics.
    Recommendations are tiered based on skill level ranges:
    - 0 to 30: Beginner level (needs significant improvement)
    - 30 to 60: Intermediate level (moderate improvement needed)
    - 60 to 100: Advanced level (refinement focused)
    
    Args:
        pitch_accuracy (float): User's pitch accuracy score
        rhythm_stability (float): User's rhythm stability score
        gamaka_proficiency (float): User's gamaka proficiency score
        breath_control (float): User's breath control score
        raga_name (str): The name of the raga being practiced
        
    Returns:
        dict: Categorized video recommendations with title, URL, description and score
    """
    recommendations = {
        "skill_improvement": [],
        "raga_examples": [],
        "technique_tutorials": []
    }
    
    # Add raga-specific examples
    raga_search_term = raga_name.replace(' ', '+')
    recommendations["raga_examples"] = [
        {
            "title": f"Professional {raga_name} Performance",
            "url": f"https://www.youtube.com/results?search_query={raga_search_term}+performance+professional",
            "description": "Watch how professionals perform this raga to understand its nuances"
        },
        {
            "title": f"{raga_name} Tutorial",
            "url": f"https://www.youtube.com/results?search_query={raga_search_term}+tutorial+beginners",
            "description": "Learn more about this raga through detailed tutorials"
        }
    ]
    
    # Pitch accuracy recommendations - tiered by score
    if pitch_accuracy < 30:
        recommendations["skill_improvement"].append({
            "title": "Fundamental Pitch Training for Beginners",
            "url": "https://www.youtube.com/results?search_query=basic+singing+pitch+training+beginners",
            "description": "Essential exercises to develop basic pitch recognition and vocal control",
            "score": pitch_accuracy,
            "level": "beginner"
        })
    elif pitch_accuracy < 60:
        recommendations["skill_improvement"].append({
            "title": "Intermediate Pitch Accuracy Exercises",
            "url": "https://www.youtube.com/results?search_query=indian+classical+vocal+pitch+training+intermediate",
            "description": "Exercises to refine your pitch control and develop consistency in transitions",
            "score": pitch_accuracy,
            "level": "intermediate"
        })
    elif pitch_accuracy < 100:
        recommendations["skill_improvement"].append({
            "title": "Advanced Pitch Refinement Techniques",
            "url": "https://www.youtube.com/results?search_query=indian+classical+vocal+advanced+pitch+precision",
            "description": "Fine-tune your pitch accuracy with advanced exercises for professional-level performance",
            "score": pitch_accuracy,
            "level": "advanced"
        })
    
    # Rhythm stability recommendations - tiered by score
    if rhythm_stability < 30:
        recommendations["skill_improvement"].append({
            "title": "Basic Rhythm and Beat Fundamentals",
            "url": "https://www.youtube.com/results?search_query=basic+rhythm+training+beginners+indian+music",
            "description": "Learn foundational concepts of rhythm and develop basic sense of timing and beat",
            "score": rhythm_stability,
            "level": "beginner"
        })
    elif rhythm_stability < 60:
        recommendations["skill_improvement"].append({
            "title": "Intermediate Rhythm Training with Taal",
            "url": "https://www.youtube.com/results?search_query=indian+classical+music+rhythm+training+taal+intermediate",
            "description": "Develop your rhythmic precision with specific taal patterns and exercises",
            "score": rhythm_stability,
            "level": "intermediate"
        })
    elif rhythm_stability < 100:
        recommendations["skill_improvement"].append({
            "title": "Advanced Rhythmic Variations and Layakari",
            "url": "https://www.youtube.com/results?search_query=advanced+indian+classical+layakari+complex+rhythms",
            "description": "Master complex rhythm patterns and develop improvisational rhythmic skills",
            "score": rhythm_stability,
            "level": "advanced"
        })
    
    # Gamaka proficiency recommendations - tiered by score
    if gamaka_proficiency < 30:
        recommendations["skill_improvement"].append({
            "title": "Introduction to Basic Gamaka Techniques",
            "url": "https://www.youtube.com/results?search_query=beginner+gamaka+ornamentations+indian+classical",
            "description": "Learn the foundational ornamentations essential for Indian classical singing",
            "score": gamaka_proficiency,
            "level": "beginner"
        })
    elif gamaka_proficiency < 60:
        recommendations["skill_improvement"].append({
            "title": "Intermediate Gamaka Techniques for Raga Expression",
            "url": "https://www.youtube.com/results?search_query=intermediate+gamaka+techniques+indian+classical+music",
            "description": "Develop more nuanced ornamentations to bring authentic expression to your performances",
            "score": gamaka_proficiency,
            "level": "intermediate"
        })
    elif gamaka_proficiency < 100:
        recommendations["skill_improvement"].append({
            "title": "Advanced Gamaka Mastery for Gharana-Specific Styles",
            "url": "https://www.youtube.com/results?search_query=advanced+gamaka+master+class+gharana+specific",
            "description": "Perfect your gamaka techniques with advanced, style-specific ornamentations used by masters",
            "score": gamaka_proficiency,
            "level": "advanced"
        })
    
    # Breath control recommendations - tiered by score
    if breath_control < 30:
        recommendations["skill_improvement"].append({
            "title": "Essential Breathing Exercises for Singers",
            "url": "https://www.youtube.com/results?search_query=breathing+basics+singing+beginners",
            "description": "Learn fundamental breathing techniques to develop proper breath support for singing",
            "score": breath_control,
            "level": "beginner"
        })
    elif breath_control < 60:
        recommendations["skill_improvement"].append({
            "title": "Intermediate Breath Control for Longer Phrases",
            "url": "https://www.youtube.com/results?search_query=breath+control+exercises+intermediate+classical+singing",
            "description": "Develop your breath capacity and control for sustaining longer musical phrases",
            "score": breath_control,
            "level": "intermediate"
        })
    elif breath_control < 100:
        recommendations["skill_improvement"].append({
            "title": "Advanced Breath Management for Complex Passages",
            "url": "https://www.youtube.com/results?search_query=advanced+breath+control+taans+classical+singing",
            "description": "Master breath techniques for performing fast taans and complex ornamentations fluidly",
            "score": breath_control,
            "level": "advanced"
        })
    
    # Determine overall skill level for general technique tutorials
    avg_score = (pitch_accuracy + rhythm_stability + gamaka_proficiency + breath_control) / 4
    
    if avg_score < 30:
        level = "beginner"
        recommendations["technique_tutorials"] = [
            {
                "title": "Foundations of Indian Classical Vocal Music",
                "url": "https://www.youtube.com/results?search_query=beginners+guide+indian+classical+vocal+basics",
                "description": "Build a strong foundation with these essential lessons for beginners",
                "level": "beginner"
            }
        ]
    elif avg_score < 60:
        level = "intermediate"
        recommendations["technique_tutorials"] = [
            {
                "title": "Intermediate Indian Classical Vocal Techniques",
                "url": "https://www.youtube.com/results?search_query=intermediate+indian+classical+vocal+techniques",
                "description": "Expand your skills with these intermediate-level technique tutorials",
                "level": "intermediate"
            }
        ]
    else:
        level = "advanced"
        recommendations["technique_tutorials"] = [
            {
                "title": "Advanced Classical Vocal Masterclass",
                "url": "https://www.youtube.com/results?search_query=advanced+indian+classical+vocal+masterclass",
                "description": "Refine your artistry with techniques from renowned classical vocalists",
                "level": "advanced"
            }
        ]
    
    return recommendations

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)