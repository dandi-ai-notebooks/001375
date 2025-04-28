# explore/plot_time_series.py
# This script loads the NWB file and plots a small segment of the time_series data

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the time_series data
time_series = nwb.acquisition["time_series"]
data = time_series.data
rate = time_series.rate

# Select a small segment of data (e.g., first 1 second from channel 0)
start_time = 0  # seconds
end_time = 1  # seconds
start_index = int(start_time * rate)
end_index = int(end_time * rate)
channel_index = 0

segment_data = data[start_index:end_index, channel_index]
time = np.linspace(start_time, end_time, len(segment_data))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time, segment_data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title(f"Time Series Data - Channel {channel_index}, Time {start_time}-{end_time}s")
plt.savefig("explore/time_series_plot.png")
plt.close()