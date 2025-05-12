# This script analyzes the trial structure in the dataset
# to understand the experimental paradigm

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trials data
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']

# Print basic statistics
print("\nTrial duration statistics (seconds):")
print(f"Mean: {trials_df['duration'].mean():.2f}")
print(f"Median: {trials_df['duration'].median():.2f}")
print(f"Min: {trials_df['duration'].min():.2f}")
print(f"Max: {trials_df['duration'].max():.2f}")
print(f"Std: {trials_df['duration'].std():.2f}")

# Visualize trial durations
plt.figure(figsize=(12, 6))
plt.hist(trials_df['duration'], bins=30, alpha=0.7, color='blue')
plt.xlabel('Trial Duration (seconds)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations')
plt.grid(True, alpha=0.3)
plt.savefig('explore/trial_durations.png', dpi=300)
print("Saved trial durations histogram to explore/trial_durations.png")

# Visualize trial start times to see timing of experiment
plt.figure(figsize=(12, 6))
plt.plot(range(len(trials_df)), trials_df['start_time'], marker='o', 
         linestyle='-', markersize=3, alpha=0.7)
plt.xlabel('Trial Number')
plt.ylabel('Start Time (seconds)')
plt.title('Trial Start Times')
plt.grid(True, alpha=0.3)
plt.savefig('explore/trial_start_times.png', dpi=300)
print("Saved trial start times plot to explore/trial_start_times.png")

# Calculate inter-trial intervals
trials_df['iti'] = trials_df['start_time'].diff()
trials_df = trials_df.dropna()  # Remove first row which has NaN ITI

print("\nInter-trial interval statistics (seconds):")
print(f"Mean: {trials_df['iti'].mean():.2f}")
print(f"Median: {trials_df['iti'].median():.2f}")
print(f"Min: {trials_df['iti'].min():.2f}")
print(f"Max: {trials_df['iti'].max():.2f}")

# Visualize inter-trial intervals
plt.figure(figsize=(12, 6))
plt.hist(trials_df['iti'], bins=30, alpha=0.7, color='green')
plt.xlabel('Inter-Trial Interval (seconds)')
plt.ylabel('Count')
plt.title('Distribution of Inter-Trial Intervals')
plt.grid(True, alpha=0.3)
plt.savefig('explore/inter_trial_intervals.png', dpi=300)
print("Saved inter-trial intervals histogram to explore/inter_trial_intervals.png")

# Now do a similar analysis for the second NWB file to compare
print("\n\nAnalyzing second NWB file for comparison...\n")
url2 = "https://api.dandiarchive.org/api/assets/376dc673-611b-4521-b135-9ec01c7f4f74/download/"
print("Loading second NWB file from:", url2)
remote_file2 = remfile.File(url2)
h5_file2 = h5py.File(remote_file2)
io2 = pynwb.NWBHDF5IO(file=h5_file2)
nwb2 = io2.read()

# Get trials data for second file
trials_df2 = nwb2.trials.to_dataframe()
print(f"Number of trials in second file: {len(trials_df2)}")

# Calculate trial durations for second file
trials_df2['duration'] = trials_df2['stop_time'] - trials_df2['start_time']

# Print basic statistics for comparison
print("\nTrial duration statistics for second file (seconds):")
print(f"Mean: {trials_df2['duration'].mean():.2f}")
print(f"Median: {trials_df2['duration'].median():.2f}")
print(f"Min: {trials_df2['duration'].min():.2f}")
print(f"Max: {trials_df2['duration'].max():.2f}")
print(f"Std: {trials_df2['duration'].std():.2f}")