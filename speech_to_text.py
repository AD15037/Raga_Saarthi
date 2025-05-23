import os
import whisper
import tempfile
import subprocess
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def optimize_audio(audio_file_path):
    """Optimize audio for better transcription"""
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    try:
        subprocess.call(['ffmpeg', '-y', '-i', audio_file_path, 
                       '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', temp_wav])
        return temp_wav
    except Exception as e:
        print(f"Conversion error: {e}")
        return audio_file_path

def speech_to_text(audio_file_path=None):
    """
    Convert speech to text using Whisper with transliteration (not translation)
    
    Parameters:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: The transcribed Hindi lyrics in English characters
    """
    if audio_file_path is None:
        print("Error: Whisper requires an audio file path")
        return ""
        
    try:
        print(f"Loading Whisper model...")
        # Use "large" for best quality 
        model = whisper.load_model("large")
        
        print(f"Processing audio file '{audio_file_path}'...")
        # Optimize audio before transcribing
        optimized_path = optimize_audio(audio_file_path)
        
        # KEY CHANGE: Use "transcribe" task with Hindi language for accurate transcription
        result = model.transcribe(
            optimized_path,
            task="transcribe",   # Just transcribe, don't translate
            language="hi",       # Specify Hindi language
            fp16=False
        )
        
        # Get the original Hindi text
        hindi_text = result["text"]
        print(f"Hindi Transcription: {hindi_text}")
        
        # Transliterate the Hindi to English characters
        transliterated_text = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
        print(f"Transliterated Text: {transliterated_text}")
        
        return transliterated_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

if __name__ == "__main__":
    text = speech_to_text("Sindhura.wav")
    print(f"Final transcription: {text}")



# Some Theory Below:
# Explanation of Mixed Capital/Lowercase Letters in ITRANS Transliteration
# The mix of capital and lowercase letters in your transliterated output is not an 
# error but a feature of the ITRANS (Indian languages TRANSliteration) scheme. 
# Here's why it happens:

# ITRANS Capitalization Rules
# ITRANS uses capitalization to represent certain Sanskrit/Hindi sounds that don't 
# exist in English like below are the examples:
# Capital A (A): represents long "aa" sound (आ) as in "father"
# Capital I (I): represents long "ee" sound (ई) as in "seen"
# Capital U (U): represents long "oo" sound (ऊ) as in "boot"
# Capital R (R): represents the Sanskrit vowel "ऋ"
# Capital T/D/N (T, D, N): represent retroflex consonants (ट, ड, ण)
# Capital S (S): represents palatal "श" (sh sound)

# Examples(Sindhura.mp3) from your text:
# pyArI = प्यारी (long 'a' sound)
# rAja = राज (long 'a' sound)
# nAda = नाद (long 'a' sound)
# bajAye = बजाये (long 'a' sound)
# AIre = आईरे (long 'a' and 'i' sounds)
