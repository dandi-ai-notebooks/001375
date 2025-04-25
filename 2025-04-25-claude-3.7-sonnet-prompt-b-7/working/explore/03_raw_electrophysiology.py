"""
This script explores the raw electrophysiology data in the first NWB file.
It visualizes signal patterns across different electrodes and calculates basic
signal statistics to better understand the data characteristics.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Set the plotting style for consistent visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)

# Create a directory for images if it doesn't exist
if not os.path.exists('explore/images'):
    os.makedirs('explore/images')

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get time series data information
time_series = nwb.acquisition["time_series"]
print(f"Time series data shape: {time_series.data.shape}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Data unit: {time_series.unit}")

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")

# Extract a small segment of data for visualization (1 second from the start)
# Note: Since the data is remote, we only load a small segment to avoid long loading times
start_time = time.time()
print("Loading 1 second of data from all channels...")
sample_rate = time_series.rate
num_samples = int(sample_rate)  # 1 second of data
data_segment = time_series.data[0:num_samples, :]
print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Calculate basic statistics for each channel
print("\nCalculating statistics for each channel...")
mean_per_channel = np.mean(data_segment, axis=0)
std_per_channel = np.std(data_segment, axis=0)
min_per_channel = np.min(data_segment, axis=0)
max_per_channel = np.max(data_segment, axis=0)
range_per_channel = max_per_channel - min_per_channel

print(f"Mean signal range across channels: {np.mean(range_per_channel):.2f} {time_series.unit}")
print(f"Mean signal std across channels: {np.mean(std_per_channel):.2f} {time_series.unit}")

# Plot statistics across channels
plt.figure(figsize=(14, 6))
plt.plot(range_per_channel, 'o-', alpha=0.5, label='Signal Range')
plt.plot(std_per_channel * 5, 'o-', alpha=0.5, label='Std Dev (x5)')
plt.xlabel('Channel Number')
plt.ylabel('Value (mV)')
plt.title('Signal Statistics by Channel (1 second sample)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('explore/images/channel_statistics.png')

# Visualize signal trace for 4 random channels from different shanks
plt.figure(figsize=(14, 10))

# Select 2 channels from each shank
shank1_channels = electrodes_df[electrodes_df['group_name'] == 'shank1'].index.values
shank2_channels = electrodes_df[electrodes_df['group_name'] == 'shank2'].index.values

np.random.seed(42)  # For reproducibility
selected_channels_shank1 = np.random.choice(shank1_channels, 2, replace=False)
selected_channels_shank2 = np.random.choice(shank2_channels, 2, replace=False)
selected_channels = np.concatenate([selected_channels_shank1, selected_channels_shank2])

# Create time vector for x-axis (100 ms)
time_vector = np.arange(0, 0.1, 1/sample_rate)
samples_to_plot = len(time_vector)

# Plot selected channels
for i, channel in enumerate(selected_channels):
    plt.subplot(4, 1, i+1)
    plt.plot(time_vector, data_segment[:samples_to_plot, channel])
    plt.title(f"Channel {channel} (Shank {'1' if channel in shank1_channels else '2'})")
    plt.ylabel('Voltage (mV)')
    plt.grid(True, alpha=0.3)

plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('explore/images/raw_traces.png')

# Plot power spectrum for the same channels
plt.figure(figsize=(14, 10))

for i, channel in enumerate(selected_channels):
    plt.subplot(4, 1, i+1)
    
    # Calculate power spectrum (use entire 1-second segment)
    signal = data_segment[:, channel]
    ps = np.abs(np.fft.rfft(signal))**2
    freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
    
    # Plot only up to 1000 Hz for better visualization
    mask = freqs <= 1000
    plt.loglog(freqs[mask], ps[mask])
    plt.title(f"Channel {channel} Power Spectrum (Shank {'1' if channel in shank1_channels else '2'})")
    plt.ylabel('Power')
    plt.grid(True, alpha=0.3)

plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('explore/images/power_spectrum.png')

# Create a heatmap of correlations between channels
# Sample a subset of channels (20) to make computation manageable
np.random.seed(42)
subset_channels = np.random.choice(range(time_series.data.shape[1]), 20, replace=False)
subset_channels.sort()  # Sort for better visualization

# Calculate correlation matrix
print("\nCalculating correlation matrix for channel subset...")
correlation_matrix = np.corrcoef(data_segment[:, subset_channels].T)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Correlation Between Channels')
plt.xlabel('Channel Index (subset)')
plt.ylabel('Channel Index (subset)')
plt.savefig('explore/images/channel_correlations.png')

# Clean up
io.close()
print("\nScript completed successfully.")