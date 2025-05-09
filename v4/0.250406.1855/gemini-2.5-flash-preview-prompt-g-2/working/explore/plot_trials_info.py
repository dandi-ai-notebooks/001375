# This script loads trials data and plots a histogram of trial durations.

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

# Get trials data
trials_df = nwb.trials.to_dataframe()

# Calculate trial durations
trial_durations = trials_df['stop_time'] - trials_df['start_time']

# Plot histogram of trial durations
plt.figure(figsize=(10, 6))
plt.hist(trial_durations, bins=50, edgecolor='black')
plt.xlabel('Trial Duration (s)')
plt.ylabel('Frequency')
plt.title('Distribution of Trial Durations')
plt.grid(True)
plt.savefig('explore/trials_info.png')
plt.close()