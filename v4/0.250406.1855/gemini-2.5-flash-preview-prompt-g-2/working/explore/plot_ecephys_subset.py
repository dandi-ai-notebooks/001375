# This script loads a subset of raw electrophysiology data and plots it.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get time series data
time_series = nwb.acquisition["time_series"]
data = time_series.data
rate = time_series.rate
starting_time = time_series.starting_time

# Load a subset of data (first 10 seconds, first 5 channels)
num_channels_to_plot = 5
duration_to_plot = 10.0 # seconds
num_samples_to_plot = int(duration_to_plot * rate)
data_subset = data[0:num_samples_to_plot, 0:num_channels_to_plot]
t = starting_time + np.arange(num_samples_to_plot) / rate

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(num_channels_to_plot):
    plt.plot(t, data_subset[:, i] + i * 500, label=f'Channel {i}') # Offset for clarity
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (offset for clarity)')
plt.title('Subset of Raw Electrophysiology Data')
plt.legend()
plt.grid(True)
plt.savefig('explore/ecephys_subset.png')
plt.close()