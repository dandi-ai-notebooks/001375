# This script visualizes a small sample of the raw electrophysiology signal
# to understand the quality and characteristics of the neural recordings

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the raw data
time_series = nwb.acquisition['time_series']
print(f"Raw data shape: {time_series.data.shape}")  # (time_points, channels)
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Data unit: {time_series.unit}")

# Sample a small segment of data from all channels
# We'll take 0.1 seconds (3000 samples at 30 kHz) from the start
n_samples = 3000  # 0.1 seconds at 30 kHz
start_sample = 0
data_sample = time_series.data[start_sample:start_sample+n_samples, :]

# Plot a subset of channels (first 4 channels)
plt.figure(figsize=(14, 10))
channels_to_plot = 4
for i in range(channels_to_plot):
    plt.subplot(channels_to_plot, 1, i+1)
    plt.plot(np.arange(n_samples) / time_series.rate, data_sample[:, i], linewidth=0.8)
    plt.title(f'Channel {i}')
    plt.ylabel('Amplitude (mV)')
    
    # Only add x-label for the bottom subplot
    if i == channels_to_plot - 1:
        plt.xlabel('Time (s)')
    
plt.tight_layout()
plt.savefig('explore/raw_signal_sample.png', dpi=300)
print("Saved raw signal sample to explore/raw_signal_sample.png")

# Now plot a longer segment for a single channel to see patterns
channel = 0  # First channel
n_samples_long = 30000 * 3  # 3 seconds
start_sample_long = 100000  # Start from later part of recording

# Get data for this longer segment
data_long = time_series.data[start_sample_long:start_sample_long+n_samples_long, channel]

# Plot the longer segment
plt.figure(figsize=(14, 6))
plt.plot(np.arange(n_samples_long) / time_series.rate, data_long, linewidth=0.5)
plt.title(f'Channel {channel} - 3 Seconds of Recording')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid(True, alpha=0.3)
plt.savefig('explore/raw_signal_longer.png', dpi=300)
print("Saved longer raw signal to explore/raw_signal_longer.png")

# Look at the frequency content using a spectrogram for this segment
plt.figure(figsize=(14, 6))
plt.specgram(data_long, NFFT=1024, Fs=time_series.rate, noverlap=512, 
             cmap='viridis', scale='dB')
plt.title(f'Spectrogram - Channel {channel}')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')
plt.ylim(0, 500)  # Limit to 0-500 Hz for better visibility
plt.savefig('explore/signal_spectrogram.png', dpi=300)
print("Saved signal spectrogram to explore/signal_spectrogram.png")