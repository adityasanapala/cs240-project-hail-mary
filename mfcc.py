import os
import librosa
import numpy as np

# Path to the directory containing the .wav files
DATASET_PATH = 'recordings/'

# Directory to save the MFCC files as .txt
MFCC_OUTPUT_DIR = 'mfcc_txt/'

# Create output directory if it doesn't exist
os.makedirs(MFCC_OUTPUT_DIR, exist_ok=True)

# Parameters
NUM_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256

# Iterate through all wav files
for filename in os.listdir(DATASET_PATH):
    if filename.endswith('.wav'):
        file_path = os.path.join(DATASET_PATH, filename)
        
        # Load the audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Pad if needed
        if len(y) < N_FFT:
            y = np.pad(y, (0, N_FFT - len(y)))
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # Transpose to (frames x coeffs) for easier viewing in text
        mfcc = mfcc.T
        
        # Create output filename
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(MFCC_OUTPUT_DIR, output_filename)
        
        # Save as .txt
        np.savetxt(output_path, mfcc, fmt='%.6f')
print("Done")
