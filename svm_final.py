import os
import sys
import numpy as np
import librosa
from svm import DigitRecognizer

def extract_mfcc_from_wav(wav_file_path, n_mfcc=13, sr=None):
    """
    Extract MFCC features from a WAV file
    
    Parameters:
    - wav_file_path: Path to the WAV file
    - n_mfcc: Number of MFCC coefficients to extract
    - sr: Sample rate (None for native sample rate)
    
    Returns:
    - mean_mfcc: Mean MFCC features
    """
    try:
        # Load WAV file
        print(f"Loading audio file: {wav_file_path}")
        y, sr = librosa.load(wav_file_path, sr=sr)
        
        # Extract MFCC features
        print("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate mean across time (axis=1)
        mean_mfcc = np.mean(mfcc, axis=1)
        
        print(f"MFCC shape: {mfcc.shape}")
        print(f"Mean MFCC shape: {mean_mfcc.shape}")
        
        return mean_mfcc
        
    except Exception as e:
        print(f"Error extracting MFCC features: {str(e)}")
        return None

def recognize_digit_from_wav(wav_file_path, model_path="digit_svm_model.pkl", 
                            scaler_path="digit_svm_scaler.pkl", n_mfcc=13):
    """
    Complete pipeline: WAV file → MFCC → Digit recognition
    
    Parameters:
    - wav_file_path: Path to the WAV file
    - model_path: Path to the saved SVM model
    - scaler_path: Path to the saved StandardScaler
    - n_mfcc: Number of MFCC coefficients to extract
    
    Returns:
    - recognized_digit: The predicted digit
    - probabilities: Probability scores for each digit class
    """
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or scaler file not found. Please train the model first.")
        return None, None
    
    # Extract MFCC features
    mean_mfcc = extract_mfcc_from_wav(wav_file_path, n_mfcc=n_mfcc)
    if mean_mfcc is None:
        return None, None
    
    # Save mean MFCC to temporary file (optional)
    temp_mfcc_file = "temp_mfcc.txt"
    np.savetxt(temp_mfcc_file, mean_mfcc)
    print(f"Mean MFCC saved to {temp_mfcc_file}")
    
    # Flatten if needed
    if mean_mfcc.ndim > 1:
        mean_mfcc = mean_mfcc.flatten()
    
    # Load model and predict
    print("Loading model and predicting digit...")
    recognizer = DigitRecognizer()
    recognizer.load_model(model_path, scaler_path)
    
    digit, probabilities = recognizer.predict(mean_mfcc)
    
    return digit, probabilities

def main():
    if len(sys.argv) != 2:
        print("Usage: python wav_to_digit.py <wav_file_path>")
        sys.exit(1)
    
    wav_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(wav_file_path):
        print(f"Error: File {wav_file_path} not found")
        sys.exit(1)
    
    # Process WAV file and recognize digit
    digit, probabilities = recognize_digit_from_wav(wav_file_path)
    
    if digit is not None:
        print("\n" + "="*50)
        print(f"RECOGNIZED DIGIT: {digit}")
        print("="*50)
        
        # Print top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        print("\nTop 3 predictions:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Digit {idx}: {probabilities[idx]:.4f}")

if __name__ == "__main__":
    main()