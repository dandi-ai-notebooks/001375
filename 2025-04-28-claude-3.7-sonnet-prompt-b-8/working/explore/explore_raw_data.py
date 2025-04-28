# This script explores the raw electrophysiology data from the NWB file
# to understand the signal characteristics

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get information about time series data
time_series = nwb.acquisition["time_series"]
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Unit: {time_series.unit}")
print(f"Data shape: {time_series.data.shape}")
print(f"Data type: {time_series.data.dtype}")

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"\nElectrode groups:")
for group_name, count in electrodes_df['group_name'].value_counts().items():
    print(f"  {group_name}: {count} electrodes")

# Sample a small segment of data for visualization (first 30,000 samples = 1 second)
# And use every 10th channel to avoid loading too much data
sample_time = 1.0  # seconds
num_samples = int(sample_time * time_series.rate)
channel_step = 10  # Use every 10th channel

print(f"\nLoading {num_samples} samples from every {channel_step}th channel...")
raw_data_sample = time_series.data[:num_samples, ::channel_step]
num_channels_sampled = raw_data_sample.shape[1]
print(f"Loaded data shape: {raw_data_sample.shape}")

# Create a time vector
time_vector = np.arange(num_samples) / time_series.rate

# Plot the first few channels
plt.figure(figsize=(15, 10))
num_channels_to_plot = min(5, num_channels_sampled)
for i in range(num_channels_to_plot):
    plt.subplot(num_channels_to_plot, 1, i+1)
    channel_idx = i * channel_step
    plt.plot(time_vector, raw_data_sample[:, i])
    plt.title(f'Channel {i*channel_step}')
    plt.ylabel(time_series.unit)
    if i == num_channels_to_plot - 1:
        plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('explore/raw_signals.png', dpi=300)

# Plot a spectrogram for one channel
plt.figure(figsize=(12, 8))
channel_idx = 0  # First channel in our sampled data
plt.specgram(raw_data_sample[:, channel_idx], NFFT=512, Fs=time_series.rate, 
             noverlap=256, cmap='viridis')
plt.title(f'Spectrogram of Channel {channel_idx*channel_step}')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.colorbar(label='Power Spectrum (dB)')
plt.tight_layout()
plt.savefig('explore/spectrogram.png', dpi=300)

# Calculate and plot power spectrum for one channel
plt.figure(figsize=(12, 6))
channel_idx = 0  # First channel in our sampled data

# Calculate power spectrum using FFT
signal = raw_data_sample[:, channel_idx]
n = len(signal)
fft_result = np.fft.rfft(signal)
power = np.abs(fft_result)**2
freqs = np.fft.rfftfreq(n, 1/time_series.rate)

# Plot only up to 1000 Hz for better visibility
max_freq = 1000
freq_mask = freqs <= max_freq
plt.plot(freqs[freq_mask], power[freq_mask])
plt.title(f'Power Spectrum of Channel {channel_idx*channel_step}')
plt.ylabel('Power')
plt.xlabel('Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/power_spectrum.png', dpi=300)

# Plot the channels across a single timepoint to see the spatial pattern
plt.figure(figsize=(12, 6))
time_idx = 1000  # Choose a specific timepoint (arbitrary)
plt.plot(raw_data_sample[time_idx, :])
plt.title(f'Signal Across Channels at Time {time_idx/time_series.rate:.4f} s')
plt.xlabel('Channel Index (every 10th)')
plt.ylabel(time_series.unit)
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/spatial_pattern.png', dpi=300)

# Close the file
h5_file.close()

print("\nPlots saved to explore/ directory.")