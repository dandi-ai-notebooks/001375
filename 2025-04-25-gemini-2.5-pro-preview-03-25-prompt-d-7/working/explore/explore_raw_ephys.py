# Explore raw ephys data snippet from Dandiset 001375, asset ce525828-8534-4b56-9e47-d2a34d1aa897
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get raw time series data
ts = nwb.acquisition['time_series']
sampling_rate = ts.rate
print(f"Sampling rate: {sampling_rate} Hz")

# Define data snippet to load (first 1 second, first 3 channels)
num_samples_to_load = int(sampling_rate * 1) # 1 second
channel_indices = [0, 1, 2]
num_channels_to_plot = len(channel_indices)
print(f"Loading {num_samples_to_load} samples for channels {channel_indices}")

# Load data
# Convert channel indices to slice or list for h5py indexing
if all(np.diff(channel_indices) == 1): # If consecutive
    channel_slice = slice(channel_indices[0], channel_indices[-1] + 1)
else:
    channel_slice = channel_indices

data_snippet = ts.data[:num_samples_to_load, channel_slice]

# Create corresponding time vector
time_vector = np.arange(num_samples_to_load) / sampling_rate

# Plot the raw data snippet
sns.set_theme()
plt.figure(figsize=(15, 5))
offset = 0
offset_increment = np.std(data_snippet) * 3 # Add offset for visibility
for i in range(num_channels_to_plot):
    channel_index = channel_indices[i]
    # Adjust index if we used a slice and channels weren't starting from 0
    plot_index = i if isinstance(channel_slice, list) else channel_index - channel_slice.start
    plt.plot(time_vector, data_snippet[:, plot_index] + offset, label=f'Channel {channel_index}')
    offset += offset_increment

plt.title(f'Raw Ephys Data Snippet (First {num_samples_to_load / sampling_rate:.2f}s)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV) + Offset')
plt.legend(loc='upper right')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
# Improve y-axis clarity (no ticks needed if signals are offset)
plt.yticks([])
plt.ylabel('Amplitude + Offset')


# Save the plot
plt.savefig('explore/raw_ephys_snippet.png')
print("\nSaved plot to explore/raw_ephys_snippet.png")

io.close()