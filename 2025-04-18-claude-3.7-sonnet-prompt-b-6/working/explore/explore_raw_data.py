"""
This script explores the raw electrophysiology data in the MS13B NWB file.
The goal is to understand the signal characteristics and visualize the raw traces.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get information about the raw data
time_series = nwb.acquisition["time_series"]
print(f"Raw data shape: {time_series.data.shape}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Unit: {time_series.unit}")

# Extract a short segment of raw data for visualization (first 1 second)
# Using a small number of channels to avoid loading too much data
sampling_rate = time_series.rate
segment_duration = 1.0  # seconds
num_samples = int(segment_duration * sampling_rate)
num_channels_to_plot = 5  # plotting only a few channels

# Select channels from different positions (first few channels from each shank)
electrodes_df = nwb.electrodes.to_dataframe()
shank1_channels = electrodes_df[electrodes_df['group_name'] == 'shank1'].index[:num_channels_to_plot]
shank2_channels = electrodes_df[electrodes_df['group_name'] == 'shank2'].index[:num_channels_to_plot]

# Get raw data for selected channels
print(f"Loading {num_samples} samples for selected channels...")
raw_data_shank1 = time_series.data[:num_samples, shank1_channels]
raw_data_shank2 = time_series.data[:num_samples, shank2_channels]

# Calculate time vector
time_vector = np.arange(num_samples) / sampling_rate

# Plot raw traces
plt.figure(figsize=(15, 10))
# Plot shank1 channels
plt.subplot(2, 1, 1)
for i in range(min(num_channels_to_plot, len(shank1_channels))):
    channel_id = shank1_channels[i]
    # Scale and offset the trace for better visualization
    offset = i * 200  # arbitrary offset to separate traces
    plt.plot(time_vector, raw_data_shank1[:, i] + offset, label=f"Channel {channel_id}")

plt.title(f"Raw Traces - Shank 1 Channels (First {segment_duration} seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV) + offset")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot shank2 channels
plt.subplot(2, 1, 2)
for i in range(min(num_channels_to_plot, len(shank2_channels))):
    channel_id = shank2_channels[i]
    # Scale and offset the trace for better visualization
    offset = i * 200  # arbitrary offset to separate traces
    plt.plot(time_vector, raw_data_shank2[:, i] + offset, label=f"Channel {channel_id}")

plt.title(f"Raw Traces - Shank 2 Channels (First {segment_duration} seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV) + offset")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('explore/raw_traces.png')

# Calculate and plot power spectrum for one channel from each shank
plt.figure(figsize=(12, 8))
# Calculate power spectrum for one channel from each shank
for shank_idx, (shank_name, data) in enumerate([("Shank 1", raw_data_shank1[:, 0]), ("Shank 2", raw_data_shank2[:, 0])]):
    # Calculate power spectrum
    ps = np.abs(np.fft.rfft(data))**2
    freqs = np.fft.rfftfreq(len(data), 1/sampling_rate)
    
    # Plot power spectrum up to 1000 Hz (typical range for neural data)
    max_freq_idx = np.searchsorted(freqs, 1000)
    plt.subplot(1, 2, shank_idx + 1)
    plt.semilogy(freqs[:max_freq_idx], ps[:max_freq_idx])
    plt.title(f"Power Spectrum - {shank_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True)

plt.tight_layout()
plt.savefig('explore/power_spectrum.png')

# Calculate and plot spectrogram for one channel from each shank
plt.figure(figsize=(15, 10))
for shank_idx, (shank_name, data) in enumerate([("Shank 1", raw_data_shank1[:, 0]), ("Shank 2", raw_data_shank2[:, 0])]):
    plt.subplot(1, 2, shank_idx + 1)
    
    # Calculate spectrogram
    segment_len = int(0.05 * sampling_rate)  # 50 ms window
    noverlap = segment_len // 2
    
    plt.specgram(data, NFFT=segment_len, Fs=sampling_rate, noverlap=noverlap, 
                cmap='viridis', scale='dB', vmin=-20, vmax=30)
    
    plt.title(f"Spectrogram - {shank_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.ylim(0, 500)  # Limit frequency range to 500 Hz for better visualization

plt.tight_layout()
plt.savefig('explore/spectrogram.png')

# Close the file
h5_file.close()