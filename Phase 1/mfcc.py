import os
import librosa
import numpy as np

# Path to the directory containing the .wav files
DATASET_PATH = './recordings/'

# Directory to save the full MFCC matrices
MFCC_OUTPUT_DIR = 'mfcc_txt/'

# Directory to save the fixed-length mean MFCC vectors
MFCC_MEAN_OUTPUT_DIR = 'mfcc_mean/'

# Create output directories if they don't exist
os.makedirs(MFCC_OUTPUT_DIR, exist_ok=True)
os.makedirs(MFCC_MEAN_OUTPUT_DIR, exist_ok=True)

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

        # Transpose to (frames x coeffs) for full MFCC matrix
        mfcc_transposed = mfcc.T

        # Save the full MFCC matrix
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(MFCC_OUTPUT_DIR, output_filename)
        np.savetxt(output_path, mfcc_transposed, fmt='%.6f')

        # Compute mean across time frames (axis=1 gives mean for each coefficient)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Save the fixed-length feature vector
        mean_output_path = os.path.join(MFCC_MEAN_OUTPUT_DIR, output_filename)
        np.savetxt(mean_output_path, mfcc_mean.reshape(1, -1), fmt='%.6f')  # save as single row

print("Done")
