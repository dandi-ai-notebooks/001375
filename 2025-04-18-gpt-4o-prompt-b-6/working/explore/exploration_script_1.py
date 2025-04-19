# Explore the NWB file's metadata and time-series data
# The goal is to generate a plot of the signal from the electrode group 'shank1'

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Set up plot style
plt.style.use('ggplot')

# Load the NWB file from the DANDI API
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb_file = io.read()

# Access the time-series data
data = nwb_file.acquisition["time_series"].data
rate = nwb_file.acquisition["time_series"].rate

# Select a subset of data to plot: 10 seconds from the second channel
num_samples = int(10 * rate)  # 10 seconds
channel_data = data[0:num_samples, 1]

# Create a time vector
time_vector = np.arange(num_samples) / rate

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(time_vector, channel_data)
plt.title('Time-Series Data (Channel 2, First 10 Seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Signal (mV)')
plt.savefig('explore/timeseries_plot.png')
plt.close()

# Close the file
io.close()