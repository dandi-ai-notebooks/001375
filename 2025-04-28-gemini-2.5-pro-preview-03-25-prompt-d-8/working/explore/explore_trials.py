# Explore trials data
# Load the trials data and print the first few rows.
# Create a histogram of trial durations.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print('Loading NWB file...')
# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print('Accessing trials data...')
trials_df = nwb.trials.to_dataframe()
print('Trials DataFrame head:')
print(trials_df.head())

print('Calculating trial durations...')
# Calculate durations
durations = trials_df['stop_time'] - trials_df['start_time']

print('Plotting histogram of trial durations...')
# Plot histogram
sns.set_theme()
plt.figure()
plt.hist(durations, bins=20)
plt.xlabel('Trial Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations')
plt.savefig('explore/trial_durations_hist.png')
plt.close() # Close the plot to prevent hanging

print('Script finished.')