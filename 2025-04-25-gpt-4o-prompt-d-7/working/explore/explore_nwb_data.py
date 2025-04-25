# This script explores and visualizes data from the NWB file of Dandiset 001375.
# Key aspects visualized include electrode positions, time series data segments, and trial intervals.

import matplotlib.pyplot as plt
import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Load NWB file using the provided link
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Visualize electrode locations
electrodes_df = nwb.electrodes.to_dataframe()
plt.figure(figsize=(10, 8))
plt.scatter(electrodes_df['x'], electrodes_df['y'], c='blue')
plt.title('Electrode Locations')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.savefig('explore/electrode_locations.png')

# Exploring a segment of time series data for visualization
time_series = nwb.acquisition['time_series']
data_segment = time_series.data[0:1000, 0]  # Small subset for visualization
plt.figure(figsize=(10, 4))
plt.plot(data_segment)
plt.title('Time Series Data Segment')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (mV)')
plt.savefig('explore/time_series_segment.png')

# Convert trial intervals to dataframe and plot trial start times
trials_df = nwb.trials.to_dataframe()
plt.figure(figsize=(10, 4))
plt.hist(trials_df['start_time'], bins=50, color='green', alpha=0.7)
plt.title('Histogram of Trial Start Times')
plt.xlabel('Start Time')
plt.ylabel('Frequency')
plt.savefig('explore/trial_start_times.png')

# Close NWB file
io.close()
remote_file.close()