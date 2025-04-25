# Explore trial durations from Dandiset 001375, asset ce525828-8534-4b56-9e47-d2a34d1aa897
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial data
trials = nwb.trials.to_dataframe()

# Calculate trial durations
trials['duration'] = trials['stop_time'] - trials['start_time']

# Plot histogram of trial durations
sns.set_theme()
plt.figure(figsize=(10, 6))
sns.histplot(trials['duration'], bins=30)
plt.title('Distribution of Trial Durations')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.grid(True)

# Save the plot
plt.savefig('explore/trial_durations.png')
print("Saved plot to explore/trial_durations.png")

# Print summary statistics
print("\nTrial Duration Statistics:")
print(trials['duration'].describe())

io.close()