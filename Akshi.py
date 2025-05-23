import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require a GUI
from flask import Flask, request, jsonify, render_template
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
from collections import Counter

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

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
RAGA_RULES_PATH = "ragas_avroh.csv"
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

# NEW: Detect gamakas (ornamentations) in the audio
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

# NEW: Create pitch histogram
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

# NEW: Detect Vadi (dominant) and Samvadi (sub-dominant) notes
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

# NEW: Segment audio into sections for analysis
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

# NEW: Analyze audio segments to find the most representative raga section
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
    
    return segment_results

# Extract features for CNN model and MATLAB-like analysis
def extract_features_from_file(file_path):
    # Load audio
    y_audio, sr = librosa.load(file_path, sr=22050)
    
    # Extract features for ML model if available
    if model_loaded:
        # Get raw pitch for rule-based matching
        pitch = librosa.yin(y_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        
        # Compute mel spectrogram for CNN
        mel_spec = librosa.feature.melspectrogram(
            y=y_audio,
            sr=sr,
            n_fft=feature_params["n_fft"],
            hop_length=feature_params["hop_length"],
            n_mels=feature_params["n_mels"]
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure consistent dimensions
        if log_mel_spec.shape[1] > feature_params["fixed_length"]:
            log_mel_spec = log_mel_spec[:, :feature_params["fixed_length"]]
        elif log_mel_spec.shape[1] < feature_params["fixed_length"]:
            # Pad with zeros if shorter than expected
            padding = feature_params["fixed_length"] - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)), mode='constant')
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
    waveform_path = f"static/graphs/waveform_{graph_id}.png"
    plt.savefig(waveform_path)
    plt.close()
    
    # 2. Plot spectrogram
    plt.figure(figsize=(10, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = f"static/graphs/spectrogram_{graph_id}.png"
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
    periodogram1_path = f"static/graphs/periodogram1_{graph_id}.png"
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
    spectrum_path = f"static/graphs/spectrum_{graph_id}.png"
    plt.savefig(spectrum_path)
    plt.close()
    
    # 5. Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_audio, bins=100)
    plt.grid(True)
    plt.xlabel('Signal Amplitude')
    plt.ylabel('Number of Samples')
    plt.title('Probability Distribution / Histogram')
    histogram_path = f"static/graphs/histogram_{graph_id}.png"
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
    autocorr_path = f"static/graphs/autocorr_{graph_id}.png"
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
    fft_path = f"static/graphs/fft_plot_{graph_id}.png"
    plt.savefig(fft_path)
    plt.close()
    
    # 8. Power spectral density using Welch's method
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_welch, pxx_welch)
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.title('Power Spectral Density - Welch Method')
    psd_path = f"static/graphs/psd_{graph_id}.png"
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
    norm_periodogram_path = f"static/graphs/norm_periodogram_{graph_id}.png"
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
    pitch_hist_path = f"static/graphs/pitch_histogram_{graph_id}.png"
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
    pitch_contour_path = f"static/graphs/pitch_contour_{graph_id}.png"
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
            "waveform": waveform_path,
            "spectrogram": spectrogram_path,
            "periodogram": periodogram1_path,
            "spectrum": spectrum_path,
            "histogram": histogram_path,
            "autocorrelation": autocorr_path,
            "fft_plot": fft_path,
            "psd": psd_path,
            "normalized_periodogram": norm_periodogram_path,
            "pitch_histogram": pitch_hist_path,  # NEW
            "pitch_contour": pitch_contour_path  # NEW
        }
    }
    
    return analysis_results

# Main route - serve the UI
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# API route for audio analysis
@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files["audio"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Save temporary file
    temp_path = os.path.join("temp", f"{uuid.uuid4().hex}.wav")
    file.save(temp_path)
    
    try:
        # Extract features for both ML and rule-based analysis
        log_mel_spec, pitch, y_audio, sr = extract_features_from_file(temp_path)
        
        results = {}
        
        # Analyze with CNN model if available
        if model_loaded and log_mel_spec is not None:
            # Reshape for CNN input
            X = log_mel_spec.reshape(1, log_mel_spec.shape[0], log_mel_spec.shape[1], 1)
            
            # Predict raga
            predictions = model.predict(X)[0]
            top_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 predictions
            
            # Decode predictions
            raga_predictions = []
            for idx in top_indices:
                raga_name = label_encoder.inverse_transform([idx])[0]
                confidence = float(predictions[idx] * 100)
                raga_predictions.append({
                    "raga": raga_name,
                    "confidence": confidence
                })
            
            results["ml_predictions"] = raga_predictions
        
        # If pitch extraction was successful, perform rule-based analysis
        if pitch is not None and len(pitch) > 0:
            # Convert pitch to swaras
            swara_notes = improved_pitch_to_swara(pitch, sr)
            
            # Create pitch histogram
            pitch_histogram = create_pitch_histogram(pitch, sr)
            
            # Detect vadi and samvadi
            vadi_samvadi = detect_vadi_samvadi(pitch, sr)
            
            # Detect gamakas
            gamaka_features = detect_gamakas(pitch, sr)
            
            # Match raga
            if raga_rules_loaded:
                raga_name, confidence, details = find_best_matching_raga(
                    swara_notes, 
                    pitch_histogram, 
                    vadi_samvadi, 
                    gamaka_features
                )
                
                results["rule_based"] = {
                    "raga": raga_name,
                    "confidence": confidence,
                    "details": details,
                    "vadi_samvadi": vadi_samvadi,
                    "swaras_detected": ''.join(swara_notes[:50]) + ('...' if len(swara_notes) > 50 else '')
                }
            
            # Perform segment analysis
            segment_results = analyze_segments(y_audio, sr)
            results["segments"] = segment_results
        
        # Perform comprehensive audio signal analysis
        signal_analysis = analyze_audio_signal(y_audio, sr)
        results["signal_analysis"] = signal_analysis
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(results)
    
    except Exception as e:
        # Clean up on error
        try:
            os.remove(temp_path)
        except:
            pass
        
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "traceback": traceback_str
        }), 500

# Route to get detailed analysis of a specific raga
@app.route("/analyze_raga/<raga_name>", methods=["GET"])
def analyze_raga(raga_name):
    if not raga_rules_loaded:
        return jsonify({"error": "Raga rules database not available"}), 404
    
    # Find the raga in the database
    raga_info = None
    for _, row in raga_df.iterrows():
        if row['name_of_the_raag'].lower() == raga_name.lower():
            raga_info = row.to_dict()
            break
    
    if raga_info is None:
        return jsonify({"error": f"Raga '{raga_name}' not found in database"}), 404
    
    # Clean up column names
    raga_data = {}
    for key, value in raga_info.items():
        # Skip NaN values
        if isinstance(value, float) and np.isnan(value):
            continue
        
        clean_key = key.replace('_', ' ').title()
        raga_data[clean_key] = value
    
    # Get additional metadata if available
    raga_data["characteristic_phrases"] = tokenize_swaras(raga_info.get('pakad', ''))
    
    # Add time of day information if available
    time_of_day = raga_info.get('time_of_day', '')
    if time_of_day:
        raga_data["Time Of Day"] = time_of_day
    
    # Add rasa (mood) information if available
    rasa = raga_info.get('rasa', '')
    if rasa:
        raga_data["Rasa"] = rasa
    
    return jsonify(raga_data)

# Route to compare two ragas
@app.route("/compare_ragas", methods=["GET"])
def compare_ragas():
    raga1 = request.args.get('raga1', '')
    raga2 = request.args.get('raga2', '')
    
    if not raga1 or not raga2:
        return jsonify({"error": "Please provide two ragas to compare"}), 400
    
    if not raga_rules_loaded:
        return jsonify({"error": "Raga rules database not available"}), 404
    
    # Find both ragas in the database
    raga1_info = None
    raga2_info = None
    
    for _, row in raga_df.iterrows():
        if row['name_of_the_raag'].lower() == raga1.lower():
            raga1_info = row.to_dict()
        if row['name_of_the_raag'].lower() == raga2.lower():
            raga2_info = row.to_dict()
    
    if raga1_info is None:
        return jsonify({"error": f"Raga '{raga1}' not found in database"}), 404
    if raga2_info is None:
        return jsonify({"error": f"Raga '{raga2}' not found in database"}), 404
    
    # Compare aaroh-avroh
    aaroh_avroh1 = raga1_info.get('aaroh_-_avroh', '')
    aaroh_avroh2 = raga2_info.get('aaroh_-_avroh', '')
    
    # Compare pakad
    pakad1 = raga1_info.get('pakad', '')
    pakad2 = raga2_info.get('pakad', '')
    
    # Compare vadi-samvadi
    vadi1 = raga1_info.get('vadi', '')
    vadi2 = raga2_info.get('vadi', '')
    samvadi1 = raga1_info.get('samvadi', '')
    samvadi2 = raga2_info.get('samvadi', '')
    
    # Calculate similarity scores
    aaroh_avroh_similarity = match_score(
        tokenize_swaras(aaroh_avroh1), 
        tokenize_swaras(aaroh_avroh2)
    )
    
    pakad_similarity = match_score(
        tokenize_swaras(pakad1), 
        tokenize_swaras(pakad2)
    )
    
    # Vadi-samvadi similarity
    vadi_samvadi_similarity = 0
    if vadi1 == vadi2:
        vadi_samvadi_similarity += 50
    if samvadi1 == samvadi2:
        vadi_samvadi_similarity += 50
    
    # Calculate overall similarity
    overall_similarity = (
        0.5 * aaroh_avroh_similarity + 
        0.3 * pakad_similarity + 
        0.2 * vadi_samvadi_similarity
    )
    
    # Generate comparison result
    comparison = {
        "raga1": raga1,
        "raga2": raga2,
        "overall_similarity": overall_similarity,
        "details": {
            "aaroh_avroh_similarity": aaroh_avroh_similarity,
            "pakad_similarity": pakad_similarity,
            "vadi_samvadi_similarity": vadi_samvadi_similarity
        },
        "raga1_features": {
            "aaroh_avroh": aaroh_avroh1,
            "pakad": pakad1,
            "vadi": vadi1,
            "samvadi": samvadi1,
            "time_of_day": raga1_info.get('time_of_day', ''),
            "rasa": raga1_info.get('rasa', '')
        },
        "raga2_features": {
            "aaroh_avroh": aaroh_avroh2,
            "pakad": pakad2,
            "vadi": vadi2,
            "samvadi": samvadi2,
            "time_of_day": raga2_info.get('time_of_day', ''),
            "rasa": raga2_info.get('rasa', '')
        }
    }
    
    return jsonify(comparison)

# Route to get a list of all ragas
@app.route("/ragas", methods=["GET"])
def get_ragas():
    if not raga_rules_loaded:
        return jsonify({"error": "Raga rules database not available"}), 404
    
    ragas = []
    for _, row in raga_df.iterrows():
        raga_name = row['name_of_the_raag']
        # Skip if name is missing
        if pd.isna(raga_name) or not raga_name:
            continue
            
        thaat = row.get('thaat', '')
        time = row.get('time_of_day', '')
        
        ragas.append({
            "name": raga_name,
            "thaat": thaat if not pd.isna(thaat) else "",
            "time_of_day": time if not pd.isna(time) else ""
        })
    
    return jsonify({"ragas": ragas})

# Route to search ragas by criteria
@app.route("/search_ragas", methods=["GET"])
def search_ragas():
    if not raga_rules_loaded:
        return jsonify({"error": "Raga rules database not available"}), 404
    
    thaat = request.args.get('thaat', '')
    time_of_day = request.args.get('time_of_day', '')
    rasa = request.args.get('rasa', '')
    swara = request.args.get('swara', '')  # Search by prominent swara
    
    matching_ragas = []
    
    for _, row in raga_df.iterrows():
        # Skip rows with missing names
        if pd.isna(row['name_of_the_raag']) or not row['name_of_the_raag']:
            continue
            
        match = True
        
        if thaat and not pd.isna(row.get('thaat', '')) and thaat.lower() != row['thaat'].lower():
            match = False
            
        if time_of_day and not pd.isna(row.get('time_of_day', '')) and time_of_day.lower() not in row['time_of_day'].lower():
            match = False
            
        if rasa and not pd.isna(row.get('rasa', '')) and rasa.lower() not in row['rasa'].lower():
            match = False
            
        if swara:
            # Check if swara is in vadi, samvadi, or prominently in aaroh-avroh
            has_swara = False
            
            if not pd.isna(row.get('vadi', '')) and swara.upper() == row['vadi'].upper():
                has_swara = True
            elif not pd.isna(row.get('samvadi', '')) and swara.upper() == row['samvadi'].upper():
                has_swara = True
            elif not pd.isna(row.get('aaroh_-_avroh', '')):
                # Count occurrences in aaroh-avroh
                swara_count = row['aaroh_-_avroh'].upper().count(swara.upper())
                if swara_count >= 2:  # Arbitrary threshold for "prominence"
                    has_swara = True
            
            if not has_swara:
                match = False
        
        if match:
            matching_ragas.append({
                "name": row['name_of_the_raag'],
                "thaat": row.get('thaat', '') if not pd.isna(row.get('thaat', '')) else "",
                "time_of_day": row.get('time_of_day', '') if not pd.isna(row.get('time_of_day', '')) else "",
                "rasa": row.get('rasa', '') if not pd.isna(row.get('rasa', '')) else "",
                "vadi": row.get('vadi', '') if not pd.isna(row.get('vadi', '')) else "",
                "samvadi": row.get('samvadi', '') if not pd.isna(row.get('samvadi', '')) else ""
            })
    
    return jsonify({"ragas": matching_ragas})

# NEW: Route to retrieve signal processing tutorials
@app.route("/tutorials", methods=["GET"])
def get_tutorials():
    tutorial_id = request.args.get('id', None)
    
    tutorials = {
        "raga_basics": {
            "title": "Understanding Ragas in Indian Classical Music",
            "content": """
# Understanding Ragas in Indian Classical Music

A raga is a melodic framework for improvisation in Indian classical music. Each raga has specific characteristics:

1. **Swara Set (Notes)**: Each raga uses a specific set of notes from the 12 semitones.
2. **Aaroh-Avroh**: The ascending and descending patterns of notes.
3. **Pakad**: Characteristic phrases that identify the raga.
4. **Vadi-Samvadi**: The dominant and sub-dominant notes.
5. **Time of Day**: Many ragas are associated with specific times of day.
6. **Rasa (Mood)**: Each raga evokes a specific emotional response.
7. **Gamakas**: Ornamentations and microtonal inflections specific to the raga.

Our raga classifier uses both machine learning and rule-based approaches to analyze these characteristics in your audio.
            """
        },
        "signal_analysis": {
            "title": "Audio Signal Analysis Basics",
            "content": """
# Audio Signal Analysis Basics

The analysis of audio signals involves several domains:

## Time Domain Analysis
- **Waveform**: Amplitude vs. time representation of the signal
- **RMS Value**: Measure of signal power
- **Crest Factor**: Ratio of peak to RMS value
- **Dynamic Range**: Ratio between loudest and quietest parts

## Frequency Domain Analysis
- **Spectrum**: Distribution of frequency components in the signal
- **Spectrogram**: Time-varying spectrum showing how frequencies change over time
- **Spectral Centroid**: Center of mass of the spectrum
- **Spectral Bandwidth**: Width of the spectrum

## Statistical Analysis
- **Histogram**: Distribution of amplitude values
- **Kurtosis**: Measure of "peakedness" of the amplitude distribution
- **Skewness**: Measure of asymmetry of the amplitude distribution

Our analysis tool provides visualizations and metrics for all these aspects of your audio signal.
            """
        },
        "feature_extraction": {
            "title": "Feature Extraction for Music Analysis",
            "content": """
# Feature Extraction for Music Analysis

## Pitch Features
- **Pitch Contour**: The fundamental frequency over time
- **Pitch Histogram**: Distribution of notes in the piece
- **Vadi-Samvadi Detection**: Finding dominant and sub-dominant notes

## Timbral Features
- **Spectral Centroid**: Brightness of the sound
- **Spectral Flux**: How quickly the spectrum changes
- **Spectral Contrast**: Difference between peaks and valleys in the spectrum
- **MFCC**: Representation of the spectral envelope

## Melodic Features
- **Note Transitions**: Patterns in how notes follow each other
- **Gamaka Detection**: Identifying ornamentations and microtonal inflections
- **Phrase Analysis**: Identifying characteristic melodic patterns

## Temporal Features
- **Tempo**: Speed of the music
- **Rhythm Patterns**: Recurring rhythmic elements
- **Onset Detection**: Finding when new notes begin

These features serve as the foundation for both machine learning models and rule-based systems for raga classification.
            """
        }
    }
    
    if tutorial_id:
        if tutorial_id in tutorials:
            return jsonify(tutorials[tutorial_id])
        else:
            return jsonify({"error": "Tutorial not found"}), 404
    else:
        # Return list of all tutorial titles
        tutorial_list = []
        for id, tutorial in tutorials.items():
            tutorial_list.append({
                "id": id,
                "title": tutorial["title"]
            })
        return jsonify({"tutorials": tutorial_list})

# Start the app if running directly
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')