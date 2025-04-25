# Script to explore a segment of raw electrophysiology data from the NWB file
# https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/
# It loads 1 second of data for the first 5 channels and saves a plot.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print("NWB file loaded.")

# Access the raw time series data
ts = nwb.acquisition['time_series']
print(f"Accessed 'time_series' data with shape: {ts.data.shape} and rate: {ts.rate} Hz")

# Define parameters for data loading
duration_to_load = 1.0  # seconds
num_channels_to_load = 5
sampling_rate = ts.rate
num_samples_to_load = int(duration_to_load * sampling_rate)
print(f"Loading {duration_to_load} seconds ({num_samples_to_load} samples) for the first {num_channels_to_load} channels.")

# Load the data subset
# Important: Load only the required slice to avoid downloading the entire dataset
data_subset = ts.data[0:num_samples_to_load, 0:num_channels_to_load]
print(f"Loaded data subset with shape: {data_subset.shape}")

# Convert data to millivolts if necessary (check units and conversion factor)
# From nwb-file-info: ts.unit is 'mV', ts.conversion is 1.0, ts.offset is 0.0
# So, the data is already in mV.
data_subset_mv = data_subset.astype(float) # Convert to float for plotting

# Create timestamps for the loaded segment
timestamps = np.arange(num_samples_to_load) / sampling_rate

# Create and save the plot
plt.figure(figsize=(15, 8))

# Add a vertical offset for each channel for clarity
offset_scale = np.ptp(data_subset_mv[:, 0]) * 1.5 if np.ptp(data_subset_mv[:, 0]) > 0 else 1
offsets = np.arange(num_channels_to_load) * offset_scale

for i in range(num_channels_to_load):
    plt.plot(timestamps, data_subset_mv[:, i] + offsets[i], label=f'Channel {i}')

plt.title(f'Raw Ephys Data Segment (First {duration_to_load}s, First {num_channels_to_load} Channels)')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (mV + offset)')
plt.yticks(offsets, [f'Ch {i}' for i in range(num_channels_to_load)]) # Label y-axis with channel numbers
plt.grid(axis='x', alpha=0.75)
plt.margins(x=0.01, y=0.01) # Add small margins

plot_path = "explore/ephys_raw_segment.png"
plt.savefig(plot_path)
print(f"Raw data segment plot saved to: {plot_path}")

# Close the file handles
io.close()
print("File handles closed.")