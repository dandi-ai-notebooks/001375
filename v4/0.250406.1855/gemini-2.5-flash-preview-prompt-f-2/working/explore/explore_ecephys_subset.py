# This script explores a subset of the raw electrophysiology data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the electrophysiology data
ecephys_data = nwb.acquisition['time_series'].data
sampling_rate = nwb.acquisition['time_series'].rate
starting_time = nwb.acquisition['time_series'].starting_time

# Select a time window (10 seconds to 10.1 seconds)
start_time = 10.0
end_time = 10.1
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)

# Select a few channels (first 4)
channel_indices = [0, 1, 2, 3]

# Load the data subset
ecephys_subset = ecephys_data[start_index:end_index, channel_indices]

# Generate timestamps for the subset
timestamps_data = starting_time + np.arange(start_index, end_index) / sampling_rate

# Plot the data
plt.figure(figsize=(10, 6))
for i, channel_index in enumerate(channel_indices):
    plt.plot(timestamps_data, ecephys_subset[:, i], label=f'Channel {channel_index}')

plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Raw Electrophysiology Data Subset")
plt.legend()
plt.grid(True)
plt.savefig('explore/ecephys_subset_plot.png')

io.close()