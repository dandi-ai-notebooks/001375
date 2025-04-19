#!/usr/bin/env python
"""
This script explores the raw time series electrophysiology data in the NWB file.
We'll visualize a small window of the raw data to understand the signal characteristics.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract information about the time series
time_series = nwb.acquisition['time_series']
print(f"Time series shape: {time_series.data.shape}")
print(f"Time series unit: {time_series.unit}")
print(f"Sampling rate: {time_series.rate} Hz")

# Load a small window of data (just 0.1 seconds)
# Since sample rate is 30,000 Hz, 3000 samples = 0.1 seconds
start_time = 100.0  # seconds into the recording
window_size = 0.1  # seconds
samples_to_load = int(window_size * time_series.rate)
start_sample = int(start_time * time_series.rate)

print(f"\nLoading data from time {start_time} to {start_time + window_size} seconds")
print(f"Sample range: {start_sample} to {start_sample + samples_to_load}")

raw_data = time_series.data[start_sample:start_sample + samples_to_load, :]
print(f"Loaded data shape: {raw_data.shape}")

# Select a few channels to plot
channels_to_plot = [0, 1, 2, 3]  # First few channels
time_vector = np.arange(raw_data.shape[0]) / time_series.rate + start_time

# Plot the raw data for these channels
plt.figure(figsize=(12, 8))
for i, channel in enumerate(channels_to_plot):
    # Offset each channel for visibility
    offset = i * 200  # offset in microvolts
    plt.plot(time_vector, raw_data[:, channel] + offset, label=f"Channel {channel}")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV) + offset")
plt.title("Raw Electrophysiology Signal")
plt.legend()
plt.savefig("explore/raw_timeseries.png")

# Plot data for a single electrode in isolation for clearer view
single_channel = 0
plt.figure(figsize=(12, 4))
plt.plot(time_vector, raw_data[:, single_channel])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.title(f"Raw Signal from Channel {single_channel}")
plt.savefig("explore/single_channel_raw.png")

# Get information about the electrodes
electrodes_df = nwb.electrodes.to_dataframe()

# Explore signals from different electrode groups
shank1_electrode_ids = electrodes_df[electrodes_df['group_name'] == 'shank1'].index.tolist()
shank2_electrode_ids = electrodes_df[electrodes_df['group_name'] == 'shank2'].index.tolist()

# Select a few electrodes from each shank
shank1_sample = shank1_electrode_ids[:3]
shank2_sample = shank2_electrode_ids[:3]

# Plot from both shanks
plt.figure(figsize=(12, 8))
for i, channel in enumerate(shank1_sample):
    plt.plot(time_vector, raw_data[:, channel] + i * 200, label=f"Shank1-{channel}")

for i, channel in enumerate(shank2_sample):
    plt.plot(time_vector, raw_data[:, channel] + (i + len(shank1_sample)) * 200, label=f"Shank2-{channel}")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV) + offset")
plt.title("Raw Signals from Different Electrode Groups")
plt.legend()
plt.savefig("explore/shank_comparison.png")

# Close the file handles
h5_file.close()
remote_file.close()