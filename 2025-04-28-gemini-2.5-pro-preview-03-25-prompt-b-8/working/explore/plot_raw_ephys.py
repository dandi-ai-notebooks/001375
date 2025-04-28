# Explore raw electrophysiology data
# Goal: Load a small segment of raw ephys data and plot the voltage trace
# for a few channels to show its structure.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the output directory exists
os.makedirs('explore', exist_ok=True)

print("Loading NWB file...")
# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Explicitly read-only
nwb = io.read()
print("NWB file loaded.")

# Access the time series data
ts = nwb.acquisition['time_series']
fs = ts.rate  # Sampling rate
conversion_factor = ts.conversion if ts.conversion else 1.0 # Use conversion factor if available

# Define time segment and channels
duration_seconds = 1.0
num_samples = int(duration_seconds * fs)
channel_indices = [0, 1, 2]

print(f"Loading data for channels {channel_indices} for the first {duration_seconds} seconds...")
# Load data snippet (samples, channels)
# Convert data to microvolts if conversion factor and unit suggest
if ts.unit == 'volts' or ts.unit == 'V':
    # Assuming conversion to microvolts makes sense
    data_snippet = ts.data[:num_samples, channel_indices] * conversion_factor * 1e6
    y_label = 'Voltage (µV)'
elif ts.unit == 'mV':
     data_snippet = ts.data[:num_samples, channel_indices] * conversion_factor * 1e3
     y_label = 'Voltage (µV)'
else:
     # Apply conversion but keep original unit if unsure
     data_snippet = ts.data[:num_samples, channel_indices] * conversion_factor
     y_label = f'Amplitude ({ts.unit})'

print("Data loaded. Shape:", data_snippet.shape)

# Create time vector
time_vector = np.arange(num_samples) / fs

print("Generating plot...")
# Plot the data
fig, ax = plt.subplots(figsize=(12, 6))
for i, ch_index in enumerate(channel_indices):
    # Offset traces for visibility
    offset = i * np.std(data_snippet[:, i]) * 5 # Adjust offset based on std dev
    ax.plot(time_vector, data_snippet[:, i] + offset, label=f'Channel {ch_index}')

ax.set_xlabel('Time (s)')
ax.set_ylabel(y_label)
ax.set_title(f'Raw Ephys Data Snippet ({duration_seconds}s, First {len(channel_indices)} Channels)')
ax.legend(loc='upper right')
ax.grid(True)
# Improve y-axis formatting - remove constant offset visualization
ax.get_yaxis().set_ticks([])

# Save the plot
output_path = 'explore/raw_ephys_snippet.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# Close resources
io.close() # Important to close the IO object
# remote_file.close() # remfile doesn't have an explicit close typically managed by gc

print("Script finished.")