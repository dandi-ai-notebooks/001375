# This script explores trial data from the NWB file
# to better understand the experimental structure

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial data
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
print("\nDescriptive statistics of trial durations (seconds):")
print(trials_df['duration'].describe())

# Plot trial durations
plt.figure(figsize=(10, 6))
plt.plot(trials_df.index, trials_df['duration'], '-o')
plt.xlabel('Trial Number')
plt.ylabel('Duration (seconds)')
plt.title('Trial Durations')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/trial_durations.png', dpi=300)

# Plot trial duration histogram
plt.figure(figsize=(10, 6))
plt.hist(trials_df['duration'], bins=20)
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.title('Histogram of Trial Durations')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/trial_durations_histogram.png', dpi=300)

# Plot trial start times
plt.figure(figsize=(10, 6))
plt.plot(trials_df.index, trials_df['start_time'] / 60, '-o')  # Convert to minutes for better readability
plt.xlabel('Trial Number')
plt.ylabel('Start Time (minutes)')
plt.title('Trial Start Times')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/trial_start_times.png', dpi=300)

# Calculate inter-trial intervals
trials_df['next_start'] = trials_df['start_time'].shift(-1)
trials_df['iti'] = trials_df['next_start'] - trials_df['stop_time']
trials_df = trials_df.dropna()  # Drop the last row which has NaN for next_start

print("\nDescriptive statistics of inter-trial intervals (seconds):")
print(trials_df['iti'].describe())

# Plot inter-trial intervals
plt.figure(figsize=(10, 6))
plt.plot(trials_df.index, trials_df['iti'], '-o')
plt.xlabel('Trial Number')
plt.ylabel('Inter-Trial Interval (seconds)')
plt.title('Inter-Trial Intervals')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/inter_trial_intervals.png', dpi=300)

# Close the file
h5_file.close()

print("\nPlots saved to explore/ directory.")