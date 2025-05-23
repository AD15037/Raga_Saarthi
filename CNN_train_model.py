import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import uuid
import csv

# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("static", exist_ok=True)

def load_raga_metadata(csv_path='Dataset.csv'):
    """Load raga metadata from CSV file into a dictionary"""
    print(f"Loading raga metadata from {csv_path}")
    raga_metadata = {}
    
    try:
        # Read CSV with proper encoding - sometimes UTF-8 with BOM is needed
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raga_name = row['Name of the raag'].strip()
                raga_metadata[raga_name] = {
                    'aaroh_avroh': row['Aaroh - Avroh'],
                    'pakad': row['Pakad'],
                    'vadi_samvadi': row['Vadi-Samvadi'],
                    'time': row['Time'],
                    'swara_set': row['Swara - Set'],
                    'gamak': row['Gamak'],
                    'rasa': row.get('Rasa(emotional essence or mood))', '')
                }
        print(f"Loaded metadata for {len(raga_metadata)} ragas")
    except Exception as e:
        print(f"Error loading raga metadata: {e}")
        raga_metadata = {}
    
    return raga_metadata

# Define feature extraction parameters (matching cnn.py)
feature_params = {
    "sample_rate": 22050,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fixed_length": 130  # Adjust based on your dataset's average length
}

# Save feature parameters
with open("models/feature_params.pkl", "wb") as f:
    pickle.dump(feature_params, f)

# Function to extract features matching cnn.py approach
def features_extractor(file):
    try:
        # Original MFCC extraction from notebook for model compatibility
        audio, sample_rate = librosa.load(file, sr=None, duration=30)  # Limit to 30 seconds
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        return mfccs_scaled_features
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

# Load dataset
def load_dataset(folder_path='Raga Recordings'):
    print(f"Loading data from {folder_path}")
    file_paths = []
    raga_names = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_paths.append(os.path.join(root, file))
                # Get raga name from directory name
                raga_name = os.path.basename(root)
                raga_names.append(raga_name)
    
    print(f"Found {len(file_paths)} audio files across {len(set(raga_names))} ragas")
    return file_paths, raga_names

def process_features(file_paths, raga_names):
    print("Extracting features...")
    extracted_features = []
    for file_path, raga_name in zip(file_paths, raga_names):
        features = features_extractor(file_path)
        if features is not None:
            extracted_features.append([features, raga_name])
    
    # Convert to DataFrame
    features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
    
    # Prepare data for training
    X = np.array(features_df['feature'].tolist())
    y = np.array(features_df['class'].tolist())
    
    # Label encoding
    le = LabelEncoder()
    encoded_y = le.fit_transform(y)
    y_categorical = to_categorical(encoded_y)
    
    # Save label encoder for prediction
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Print class information
    print("Original labels:", le.classes_)
    print("Number of classes:", len(le.classes_))
    
    return X, y_categorical, le

def train_model(X, y):
    print("Training model...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=127)
    
    # Flatten features for 1D CNN
    X_train_flat = np.array([x.flatten() for x in X_train])
    X_test_flat = np.array([x.flatten() for x in X_test])
    
    # Get number of classes
    num_labels = y.shape[1]
    
    # Build model (from notebook)
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu', 
                          input_shape=(X_train_flat.shape[1], 1)))
    model.add(layers.MaxPooling1D(pool_size=1))
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_labels, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                 metrics=['accuracy'],
                 optimizer='Adam')
    
    # Train model
    print("Model summary:")
    model.summary()
    
    num_epochs = 50
    num_batch_size = 4
    
    # Save checkpoints during training
    checkpointer = ModelCheckpoint(filepath='models/checkpoint_model.h5',
                                 verbose=1,
                                 save_best_only=True)
    
    # Train model
    start = datetime.now()
    history = model.fit(
        X_train_flat.reshape(X_train_flat.shape[0], X_train_flat.shape[1], 1), 
        y_train,
        batch_size=num_batch_size,
        epochs=num_epochs,
        validation_data=(X_test_flat.reshape(X_test_flat.shape[0], X_test_flat.shape[1], 1), y_test),
        verbose=1,
        callbacks=[checkpointer]
    )
    
    duration = datetime.now() - start
    print("Training completed in time:", duration)
    
    # Evaluate model
    test_accuracy = model.evaluate(
        X_test_flat.reshape(X_test_flat.shape[0], X_test_flat.shape[1], 1),
        y_test,
        verbose=0
    )
    train_accuracy = model.evaluate(
        X_train_flat.reshape(X_train_flat.shape[0], X_train_flat.shape[1], 1),
        y_train,
        verbose=0
    )
    
    print(f"Test set accuracy: {test_accuracy[1] * 100:.2f}%")
    print(f"Train set accuracy: {train_accuracy[1] * 100:.2f}%")
    
    # Save model in format compatible with cnn.py
    model.save("models/cnn_raga_classifier.h5")
    print("Model saved to models/cnn_raga_classifier.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/training_history.png')
    plt.close()
    
    return model, history

def test_prediction(model, le, raga_metadata, test_file):
    """Test the model with a sample file"""
    try:
        print(f"\nTesting prediction with file: {test_file}")
        prediction_feature = features_extractor(test_file)
        if prediction_feature is None:
            print("Failed to extract features from test file")
            return
            
        prediction_feature = prediction_feature.reshape(1, -1, 1)
        predicted_probabilities = model.predict(prediction_feature)
        predicted_class_label = np.argmax(predicted_probabilities)
        prediction_class = le.inverse_transform([predicted_class_label])
        predicted_raga = prediction_class[0]
        confidence = predicted_probabilities[0][predicted_class_label] * 100
        
        print(f"Predicted raga: {prediction_class[0]}")
        print(f"Confidence: {confidence:.2f}%")

        # Display raga metadata if available
        if predicted_raga in raga_metadata:
            metadata = raga_metadata[predicted_raga]
            print("\nRaga characteristics:")
            print(f"• Aaroh-Avroh: {metadata['aaroh_avroh']}")
            print(f"• Pakad: {metadata['pakad']}")
            print(f"• Vadi-Samvadi: {metadata['vadi_samvadi']}")
            print(f"• Time of day: {metadata['time']}")
            print(f"• Swara Set: {metadata['swara_set']}")
            print(f"• Emotional essence (Rasa): {metadata['rasa']}")
        else:
            print(f"No metadata available for raga: {predicted_raga}")
            
        # Save prediction results to file
        result = {
            "raga": predicted_raga,
            "confidence": confidence,
            "metadata": raga_metadata.get(predicted_raga, {})
        }
        
        # Save prediction results
        with open("temp/last_prediction.pkl", "wb") as f:
            pickle.dump(result, f)
            
        return result
    except Exception as e:
        print(f"Error testing prediction: {e}")
        return None
    
def generate_raga_report(audio_file, model=None, label_encoder=None, raga_metadata=None):
    """Generate a comprehensive analysis of an audio file, including raga prediction and metadata"""
    
    # Load model and metadata if not provided
    if model is None or label_encoder is None or raga_metadata is None:
        try:
            model = models.load_model("models/cnn_raga_classifier.h5")
            with open("models/label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            with open("models/raga_metadata.pkl", "rb") as f:
                raga_metadata = pickle.load(f)
        except Exception as e:
            print(f"Error loading model or metadata: {e}")
            return None
    
    # Predict raga
    result = test_prediction(model, label_encoder, raga_metadata, audio_file)
    if result is None:
        return None
    
    # Generate HTML report
    predicted_raga = result["raga"]
    metadata = result.get("metadata", {})
    
    html_report = f"""
    <html>
    <head>
        <title>Raga Analysis: {predicted_raga}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #4a4a4a; }}
            .confidence {{ font-size: 1.2em; margin-bottom: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .characteristic {{ margin-bottom: 10px; }}
            .label {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Raga Analysis: {predicted_raga}</h1>
        <div class="confidence">Confidence: {result["confidence"]:.2f}%</div>
        
        <div class="section">
            <h2>Raga Characteristics</h2>
            <div class="characteristic"><span class="label">Aaroh-Avroh:</span> {metadata.get('aaroh_avroh', 'N/A')}</div>
            <div class="characteristic"><span class="label">Pakad:</span> {metadata.get('pakad', 'N/A')}</div>
            <div class="characteristic"><span class="label">Vadi-Samvadi:</span> {metadata.get('vadi_samvadi', 'N/A')}</div>
            <div class="characteristic"><span class="label">Time of Day:</span> {metadata.get('time', 'N/A')}</div>
            <div class="characteristic"><span class="label">Swara Set:</span> {metadata.get('swara_set', 'N/A')}</div>
            <div class="characteristic"><span class="label">Rasa (Emotional Essence):</span> {metadata.get('rasa', 'N/A')}</div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = f"static/raga_report_{os.path.basename(audio_file).split('.')[0]}.html"
    with open(report_path, "w") as f:
        f.write(html_report)
    
    print(f"Raga analysis report generated: {report_path}")
    return report_path

def main():
    print("Starting raga classification model training...")
    
    # Load raga metadata
    raga_metadata = load_raga_metadata()
    
    # Load dataset
    file_paths, raga_names = load_dataset()
    
    # Verify ragas in dataset against metadata
    unique_ragas = set(raga_names)
    metadata_ragas = set(raga_metadata.keys())
    
    print(f"Ragas in audio dataset: {len(unique_ragas)}")
    print(f"Ragas in metadata: {len(metadata_ragas)}")
    
    # List ragas in dataset not found in metadata
    missing_metadata = unique_ragas - metadata_ragas
    if missing_metadata:
        print(f"Warning: {len(missing_metadata)} ragas in dataset have no metadata:")
        print(', '.join(missing_metadata))
    
    # Extract features
    X, y, label_encoder = process_features(file_paths, raga_names)
    
    # Train model
    model, history = train_model(X, y)
    
    # Save raga metadata with the model
    with open("models/raga_metadata.pkl", "wb") as f:
        pickle.dump(raga_metadata, f)
    
    # Test prediction if test file exists
    test_file = "bhairavi30.wav"
    if os.path.exists(test_file):
        test_prediction(model, label_encoder, raga_metadata, test_file)
    else:
        print(f"Test file {test_file} not found. Skipping prediction test.")
    
    print("Model training completed successfully.")

if __name__ == "__main__":
    main()