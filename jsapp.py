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

# Utility: extract swara tokens from a pattern string
def tokenize_swaras(swara_string):
    return [note.strip() for note in swara_string.replace('-', ' ').replace(',', ' ').split() if note.strip()]

# Utility: compare swara sequences
def match_score(input_seq, rule_seq):
    return SequenceMatcher(None, input_seq, rule_seq).ratio() * 100

# Match raga based on pakad or aaroh-avroh
def find_best_matching_raga(pitch_notes):
    if not raga_rules_loaded:
        return "Unknown", 0
    
    best_raga = None
    best_score = 0

    for _, row in raga_df.iterrows():
        raga_name = row['name_of_the_raag']
        aaroh_avroh = row.get('aaroh_-_avroh', '')
        pakad = row.get('pakad', '')

        rule_tokens = tokenize_swaras(f"{aaroh_avroh} {pakad}")
        score = match_score(pitch_notes, rule_tokens)

        if score > best_score:
            best_score = score
            best_raga = raga_name

    return best_raga, best_score

# Convert pitch values to approximate note names
def pitch_to_note_names(pitch_values):
    note_names = []
    for hz in pitch_values:
        if hz > 0:
            try:
                note = librosa.hz_to_note(hz, octave=False)
                note_names.append(note)
            except:
                # Skip invalid pitch values
                pass
    # Make sure we have at least some notes
    if not note_names:
        note_names = ["C", "D", "E"]  # Default notes if none were detected
    return note_names

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
            "periodogram": periodogram1_path,  # NEW
            "spectrum": spectrum_path,
            "histogram": histogram_path,
            "autocorrelation": autocorr_path,
            "fft_plot": fft_path,  # NEW
            "psd": psd_path,
            "normalized_periodogram": norm_periodogram_path  # NEW
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
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    original_filename = audio_file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    if file_extension not in ['.wav', '.mp3']:
        return jsonify({"error": "Unsupported audio format. Please upload .wav or .mp3 file."}), 400
    
    filename = f"temp_{uuid.uuid4().hex}{file_extension}"
    filepath = os.path.join("temp", filename)
    audio_file.save(filepath)

    try:
        # Extract features for analysis
        features, pitch_values, y_audio, sr = extract_features_from_file(filepath)
        
        # Perform MATLAB-style audio analysis
        analysis_results = analyze_audio_signal(y_audio, sr)
        
        # ML-based raga classification if model is loaded
        raga_results = {}
        if model_loaded:
            # Process features for CNN model
            features_normalized = (features - features.mean()) / (features.std() + 1e-8)
            features_reshaped = features_normalized.reshape(1, 
                                                        feature_params["n_mels"], 
                                                        feature_params["fixed_length"], 
                                                        1)
            
            try:
                # Get raw prediction probabilities
                prediction_probs = model.predict(features_reshaped)
                # Get the predicted class index
                predicted_class_idx = np.argmax(prediction_probs[0])
                # Convert to raga name using label encoder
                ml_predicted_raga = label_encoder.inverse_transform([predicted_class_idx])[0]
                # Get confidence score
                confidence = float(prediction_probs[0][predicted_class_idx] * 100)
                
                # Rule-based raga matching
                pitch_note_names = pitch_to_note_names(pitch_values)
                raga_rule_match, rule_score = find_best_matching_raga(pitch_note_names)
                
                # Performance metrics
                tone_score = np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr))
                tempo = librosa.beat.tempo(y=y_audio, sr=sr)[0]  # Access first element of the array
                rhythm_score = tempo
                try:
                    pitch_score = np.mean([p for p in pitch_values if p > 0])
                except:
                    pitch_score = 0
                
                tone_score_norm = min(tone_score / 5000 * 100, 100)
                rhythm_score_norm = min(rhythm_score / 300 * 100, 100)
                pitch_score_norm = min(pitch_score / 500 * 100, 100)
                
                # Performance visualization
                labels = ['Pitch', 'Rhythm', 'Tone', 'Rule Match']
                scores = [pitch_score_norm, rhythm_score_norm, tone_score_norm, rule_score]
                
                plt.figure(figsize=(6, 4))
                plt.bar(labels, scores, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
                plt.ylim(0, 100)
                plt.ylabel("Score (%)")
                plt.title("Vocal Performance Evaluation")
                graph_filename = f"performance_{uuid.uuid4().hex}.png"
                graph_path = os.path.join("static", graph_filename)
                plt.savefig(graph_path)
                plt.close()
                
                raga_results = {
                    "ml_predicted_raga": ml_predicted_raga,
                    "ml_confidence": confidence,
                    "rule_based_raga": raga_rule_match,
                    "rule_match_score": rule_score,
                    "pitch_score": pitch_score_norm,
                    "rhythm_score": rhythm_score_norm,
                    "tone_score": tone_score_norm,
                    "performance_graph_url": f"/{graph_path}"
                }
            except Exception as e:
                print(f"Model prediction failed: {e}")
                raga_results = {
                    "ml_predicted_raga": "Unknown",
                    "ml_confidence": 0.0,
                    "error": f"Model prediction failed: {str(e)}"
                }
        
        # Combine analysis results
        result = {
            **analysis_results,
            **raga_results
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    finally:
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)

# Create a simple HTML template
@app.route("/template", methods=["GET"])
def get_template():
    # Create templates directory first
    os.makedirs("templates", exist_ok=True)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Analysis Tool</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .upload-form { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .results { margin-top: 20px; }
            .graphs { display: flex; flex-wrap: wrap; }
            .graph-item { margin: 10px; text-align: center; }
            .graph-item img { max-width: 100%; border: 1px solid #eee; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Analysis Tool</h1>
            
            <div class="upload-form">
                <h2>Upload Audio File</h2>
                <form id="audioForm" enctype="multipart/form-data">
                    <input type="file" name="audio" accept=".wav,.mp3" required>
                    <button type="submit">Analyze</button>
                </form>
            </div>
            
            <div class="results" id="results" style="display:none;">
                <h2>Analysis Results</h2>
                
                <div id="loading" style="display:none;">
                    <p>Processing... Please wait.</p>
                </div>
                
                <div id="statsResults">
                    <h3>Signal Statistics</h3>
                    <table id="statsTable">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </table>
                </div>
                
                <div id="ragaResults" style="display:none;">
                    <h3>Raga Analysis</h3>
                    <table id="ragaTable">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </table>
                    <div id="performanceGraph" class="graph-item"></div>
                </div>
                
                <h3>Signal Visualizations</h3>
                <div class="graphs" id="graphs"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('audioForm').addEventListener('submit', function(event) {
                event.preventDefault();
                
                const formData = new FormData(this);
                const resultsDiv = document.getElementById('results');
                const loadingDiv = document.getElementById('loading');
                const graphsDiv = document.getElementById('graphs');
                const statsTable = document.getElementById('statsTable');
                const ragaTable = document.getElementById('ragaTable');
                const ragaResults = document.getElementById('ragaResults');
                
                // Reset previous results
                while (statsTable.rows.length > 1) {
                    statsTable.deleteRow(1);
                }
                while (ragaTable.rows.length > 1) {
                    ragaTable.deleteRow(1);
                }
                graphsDiv.innerHTML = '';
                document.getElementById('performanceGraph').innerHTML = '';
                
                // Show loading and results area
                resultsDiv.style.display = 'block';
                loadingDiv.style.display = 'block';
                
                fetch('/analyze', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loadingDiv.style.display = 'none';
                    
                    // Populate time domain stats
                    const timeStats = data.time_domain;
                    for (const [key, value] of Object.entries(timeStats)) {
                        const row = statsTable.insertRow();
                        const cell1 = row.insertCell(0);
                        const cell2 = row.insertCell(1);
                        cell1.textContent = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                    }
                    
                    // Populate frequency domain stats
                    const freqStats = data.frequency_domain;
                    for (const [key, value] of Object.entries(freqStats)) {
                        const row = statsTable.insertRow();
                        const cell1 = row.insertCell(0);
                        const cell2 = row.insertCell(1);
                        cell1.textContent = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                    }
                    
                    // Display graphs
                    for (const [name, path] of Object.entries(data.graphs)) {
                        const graphDiv = document.createElement('div');
                        graphDiv.className = 'graph-item';
                        
                        const title = document.createElement('h4');
                        title.textContent = name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                        
                        const img = document.createElement('img');
                        img.src = path;
                        img.alt = name;
                        
                        graphDiv.appendChild(title);
                        graphDiv.appendChild(img);
                        graphsDiv.appendChild(graphDiv);
                    }
                    
                    // Show raga results if available
                    if (data.ml_predicted_raga) {
                        ragaResults.style.display = 'block';
                        
                        // Add raga classification results
                        addRowToTable(ragaTable, "ML Predicted Raga", data.ml_predicted_raga);
                        addRowToTable(ragaTable, "ML Confidence", data.ml_confidence.toFixed(2) + "%");
                        addRowToTable(ragaTable, "Rule-Based Raga", data.rule_based_raga);
                        addRowToTable(ragaTable, "Rule Match Score", data.rule_match_score.toFixed(2) + "%");
                        addRowToTable(ragaTable, "Pitch Score", data.pitch_score.toFixed(2) + "%");
                        addRowToTable(ragaTable, "Rhythm Score", data.rhythm_score.toFixed(2) + "%");
                        addRowToTable(ragaTable, "Tone Score", data.tone_score.toFixed(2) + "%");
                        
                        // Display performance graph
                        if (data.performance_graph_url) {
                            const img = document.createElement('img');
                            img.src = data.performance_graph_url;
                            img.alt = "Performance Evaluation";
                            document.getElementById('performanceGraph').appendChild(img);
                        }
                    }
                })
                .catch(error => {
                    loadingDiv.style.display = 'none';
                    alert('Error: ' + error.message);
                });
            });
            
            function addRowToTable(table, label, value) {
                const row = table.insertRow();
                const cell1 = row.insertCell(0);
                const cell2 = row.insertCell(1);
                cell1.textContent = label;
                cell2.textContent = value;
            }
        </script>
    </body>
    </html>
    """
    
    # Create template file
    with open("templates/index.html", "w") as f:
        f.write(html_content)
    
    return html_content

# Create templates directory and index.html on startup
def create_template_file():
    os.makedirs("templates", exist_ok=True)
    with open("templates/index.html", "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audio Analysis Tool</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .upload-form { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                .results { margin-top: 20px; }
                .graphs { display: flex; flex-wrap: wrap; }
                .graph-item { margin: 10px; text-align: center; }
                .graph-item img { max-width: 100%; border: 1px solid #eee; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Audio Analysis Tool</h1>
                
                <div class="upload-form">
                    <h2>Upload Audio File</h2>
                    <form id="audioForm" enctype="multipart/form-data">
                        <input type="file" name="audio" accept=".wav,.mp3" required>
                        <button type="submit">Analyze</button>
                    </form>
                </div>
                
                <div class="results" id="results" style="display:none;">
                    <h2>Analysis Results</h2>
                    
                    <div id="loading" style="display:none;">
                        <p>Processing... Please wait.</p>
                    </div>
                    
                    <div id="statsResults">
                        <h3>Signal Statistics</h3>
                        <table id="statsTable">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </table>
                    </div>
                    
                    <div id="ragaResults" style="display:none;">
                        <h3>Raga Analysis</h3>
                        <table id="ragaTable">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </table>
                        <div id="performanceGraph" class="graph-item"></div>
                    </div>
                    
                    <h3>Signal Visualizations</h3>
                    <div class="graphs" id="graphs"></div>
                </div>
            </div>
            
            <script>
                document.getElementById('audioForm').addEventListener('submit', function(event) {
                    event.preventDefault();
                    
                    const formData = new FormData(this);
                    const resultsDiv = document.getElementById('results');
                    const loadingDiv = document.getElementById('loading');
                    const graphsDiv = document.getElementById('graphs');
                    const statsTable = document.getElementById('statsTable');
                    const ragaTable = document.getElementById('ragaTable');
                    const ragaResults = document.getElementById('ragaResults');
                    
                    // Reset previous results
                    while (statsTable.rows.length > 1) {
                        statsTable.deleteRow(1);
                    }
                    while (ragaTable.rows.length > 1) {
                        ragaTable.deleteRow(1);
                    }
                    graphsDiv.innerHTML = '';
                    document.getElementById('performanceGraph').innerHTML = '';
                    
                    // Show loading and results area
                    resultsDiv.style.display = 'block';
                    loadingDiv.style.display = 'block';
                    
                    fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        loadingDiv.style.display = 'none';
                        
                        // Populate time domain stats
                        const timeStats = data.time_domain;
                        for (const [key, value] of Object.entries(timeStats)) {
                            const row = statsTable.insertRow();
                            const cell1 = row.insertCell(0);
                            const cell2 = row.insertCell(1);
                            cell1.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                        }
                        
                        // Populate frequency domain stats
                        const freqStats = data.frequency_domain;
                        for (const [key, value] of Object.entries(freqStats)) {
                            const row = statsTable.insertRow();
                            const cell1 = row.insertCell(0);
                            const cell2 = row.insertCell(1);
                            cell1.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
                        }
                        
                        // Display graphs
                        for (const [name, path] of Object.entries(data.graphs)) {
                            const graphDiv = document.createElement('div');
                            graphDiv.className = 'graph-item';
                            
                            const title = document.createElement('h4');
                            title.textContent = name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            
                            const img = document.createElement('img');
                            img.src = path;
                            img.alt = name;
                            
                            graphDiv.appendChild(title);
                            graphDiv.appendChild(img);
                            graphsDiv.appendChild(graphDiv);
                        }
                        
                        // Show raga results if available
                        if (data.ml_predicted_raga) {
                            ragaResults.style.display = 'block';
                            
                            // Add raga classification results
                            addRowToTable(ragaTable, "ML Predicted Raga", data.ml_predicted_raga);
                            addRowToTable(ragaTable, "ML Confidence", data.ml_confidence.toFixed(2) + "%");
                            addRowToTable(ragaTable, "Rule-Based Raga", data.rule_based_raga);
                            addRowToTable(ragaTable, "Rule Match Score", data.rule_match_score.toFixed(2) + "%");
                            addRowToTable(ragaTable, "Pitch Score", data.pitch_score.toFixed(2) + "%");
                            addRowToTable(ragaTable, "Rhythm Score", data.rhythm_score.toFixed(2) + "%");
                            addRowToTable(ragaTable, "Tone Score", data.tone_score.toFixed(2) + "%");
                            
                            // Display performance graph
                            if (data.performance_graph_url) {
                                const img = document.createElement('img');
                                img.src = data.performance_graph_url;
                                img.alt = "Performance Evaluation";
                                document.getElementById('performanceGraph').appendChild(img);
                            }
                        }
                    })
                    .catch(error => {
                        loadingDiv.style.display = 'none';
                        alert('Error: ' + error.message);
                    });
                });
                
                function addRowToTable(table, label, value) {
                    const row = table.insertRow();
                    const cell1 = row.insertCell(0);
                    const cell2 = row.insertCell(1);
                    cell1.textContent = label;
                    cell2.textContent = value;
                }
            </script>
        </body>
        </html>
        """)

# Create startup code for the Flask app
if __name__ == "__main__":
    # Create the template file on startup
    create_template_file()
    # Start the Flask server
    app.run(host="0.0.0.0", debug=True)