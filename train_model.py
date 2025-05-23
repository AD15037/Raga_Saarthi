import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path to dataset directory: each subfolder should be named after a raga and contain .wav files
DATASET_PATH = "Raga Recordings"

# Function to extract features from a single audio file
def extract_features(file_path):
    try:
        y_audio, sr = librosa.load(file_path)
        
        # Get pitch and ensure it's a scalar
        pitch = librosa.yin(y_audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
        pitch_mean = float(np.mean(pitch))  # Convert to scalar float
        
        # Get tempo as scalar
        tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
        tempo = float(np.mean(tempo)) if hasattr(tempo, '__len__') else float(tempo)  # Ensure it's a scalar
        
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1).tolist()  # Convert to list
        
        # Create feature vector as a flat list first
        feature_list = [pitch_mean, tempo] + mfcc_means
        
        # Then convert to numpy array
        features = np.array(feature_list)
        
        # Check for invalid values
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"Warning: Invalid values in features for {file_path}")
            return None
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and extract features
X = []
y = []

for raga_folder in os.listdir(DATASET_PATH):
    raga_path = os.path.join(DATASET_PATH, raga_folder)
    if os.path.isdir(raga_path):
        for filename in os.listdir(raga_path):
            if filename.endswith((".wav", ".mp3")):
                file_path = os.path.join(raga_path, filename)
                features = extract_features(file_path)
                if features is not None:
                    # Append to lists
                    X.append(features)
                    y.append(raga_folder)

# Validate feature extraction
for i, feature in enumerate(X):
    if len(feature) != len(X[0]):
        print(f"Inconsistent feature length at index {i}: {len(feature)} != {len(X[0])}")
        exit(1)

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
with open("models/classifier_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model saved as models/classifier_model.pkl")