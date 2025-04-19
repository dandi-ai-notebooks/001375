"""
This script explores the trial structure in the NWB files.
The goal is to understand the behavioral aspects of the experiment,
including trial durations and patterns.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Function to load an NWB file
def load_nwb(asset_id):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    return io.read(), h5_file

# Load the NWB files
print("Loading NWB files...")
# MS13B
nwb1, h5_file1 = load_nwb("ce525828-8534-4b56-9e47-d2a34d1aa897")
# MS14A - obj-12781w8
nwb2, h5_file2 = load_nwb("376dc673-611b-4521-b135-9ec01c7f4f74")
# MS14A - obj-ardad2
nwb3, h5_file3 = load_nwb("fea95c0e-2f07-49a3-b607-4b7e9f278e16")

# Get trials information for each file
trials1 = nwb1.intervals['trials'].to_dataframe()
trials2 = nwb2.intervals['trials'].to_dataframe()
trials3 = nwb3.intervals['trials'].to_dataframe()

print(f"MS13B: {len(trials1)} trials")
print(f"MS14A (obj-12781w8): {len(trials2)} trials")
print(f"MS14A (obj-ardad2): {len(trials3)} trials")

# Calculate trial durations
durations1 = trials1['stop_time'] - trials1['start_time']
durations2 = trials2['stop_time'] - trials2['start_time']
durations3 = trials3['stop_time'] - trials3['start_time']

# Summary statistics
print("\nTrial duration summary statistics (seconds):")
print("\nMS13B:")
print(durations1.describe())
print("\nMS14A (obj-12781w8):")
print(durations2.describe())
print("\nMS14A (obj-ardad2):")
print(durations3.describe())

# Plot trial durations
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(durations1, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Trial Durations - MS13B')
plt.ylabel('Duration (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
plt.plot(durations2, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Trial Durations - MS14A (obj-12781w8)')
plt.ylabel('Duration (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
plt.plot(durations3, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Trial Durations - MS14A (obj-ardad2)')
plt.xlabel('Trial Number')
plt.ylabel('Duration (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('explore/trial_durations.png')

# Plot histograms of trial durations
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.hist(durations1, bins=30, alpha=0.7)
plt.title('Distribution of Trial Durations - MS13B')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(3, 1, 2)
plt.hist(durations2, bins=30, alpha=0.7)
plt.title('Distribution of Trial Durations - MS14A (obj-12781w8)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(3, 1, 3)
plt.hist(durations3, bins=30, alpha=0.7)
plt.title('Distribution of Trial Durations - MS14A (obj-ardad2)')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('explore/trial_duration_histograms.png')

# Plot inter-trial intervals (time between trials)
plt.figure(figsize=(15, 10))

# Calculate inter-trial intervals
iti1 = trials1['start_time'].iloc[1:].values - trials1['stop_time'].iloc[:-1].values
iti2 = trials2['start_time'].iloc[1:].values - trials2['stop_time'].iloc[:-1].values
iti3 = trials3['start_time'].iloc[1:].values - trials3['stop_time'].iloc[:-1].values

plt.subplot(3, 1, 1)
plt.plot(iti1, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Inter-trial Intervals - MS13B')
plt.ylabel('Interval (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 2)
plt.plot(iti2, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Inter-trial Intervals - MS14A (obj-12781w8)')
plt.ylabel('Interval (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(3, 1, 3)
plt.plot(iti3, marker='o', linestyle='-', markersize=3, alpha=0.7)
plt.title('Inter-trial Intervals - MS14A (obj-ardad2)')
plt.xlabel('Trial Number')
plt.ylabel('Interval (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('explore/inter_trial_intervals.png')

# Print summary of inter-trial intervals
print("\nInter-trial interval summary statistics (seconds):")
print("\nMS13B:")
print(pd.Series(iti1).describe())
print("\nMS14A (obj-12781w8):")
print(pd.Series(iti2).describe())
print("\nMS14A (obj-ardad2):")
print(pd.Series(iti3).describe())

# Close the files
h5_file1.close()
h5_file2.close()
h5_file3.close()