import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# === Set Paths ===
RECORDINGS_PATH = 'recordings/'
OUTPUT_DIR = 'aggregate_outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load All .wav Files ===
wav_files = glob.glob(os.path.join(RECORDINGS_PATH, '*.wav'))
wav_files.sort()

# === Variables to Hold Data ===
waveforms = []
mel_specs = []
durations = []
N_FFT = 512

# === Preprocessing ===
print("Loading and processing files...")
for file_path in wav_files:
    y, sr = librosa.load(file_path, sr=None)
    durations.append(librosa.get_duration(y=y, sr=sr))
    waveforms.append(y)
    
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=N_FFT, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    mel_specs.append(S_db)

# === Average Waveform ===
min_samples = min([len(y) for y in waveforms])
waveforms_aligned = [y[:min_samples] for y in waveforms]
avg_waveform = np.mean(waveforms_aligned, axis=0)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(avg_waveform, sr=sr)
plt.title('Average Waveform Across All Samples')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'average_waveform.png'))
plt.close()
print("Saved average waveform plot.")

# === Average Mel Spectrogram ===
min_frames = min([spec.shape[1] for spec in mel_specs])
mel_specs_aligned = [spec[:, :min_frames] for spec in mel_specs]
avg_mel_spec = np.mean(mel_specs_aligned, axis=0)

plt.figure(figsize=(10, 4))
librosa.display.specshow(avg_mel_spec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Average Mel Spectrogram Across All Samples')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'average_mel_spectrogram.png'))
plt.close()
print("Saved average mel spectrogram plot.")

# === Duration Histogram ===
plt.figure(figsize=(8, 4))
plt.hist(durations, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Audio Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'duration_histogram.png'))
plt.close()
print("Saved duration histogram plot.")
