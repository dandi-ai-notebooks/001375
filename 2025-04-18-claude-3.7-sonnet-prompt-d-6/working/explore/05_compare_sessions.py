#!/usr/bin/env python
"""
This script compares key properties between two NWB files from the Dandiset
to understand differences between recording sessions.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# URLs for the two NWB files we want to compare
url1 = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"  # MS13B
url2 = "https://api.dandiarchive.org/api/assets/fea95c0e-2f07-49a3-b607-4b7e9f278e16/download/"  # MS14A

print("Loading first NWB file (MS13B)...")
remote_file1 = remfile.File(url1)
h5_file1 = h5py.File(remote_file1)
io1 = pynwb.NWBHDF5IO(file=h5_file1)
nwb1 = io1.read()

print("Loading second NWB file (MS14A)...")
remote_file2 = remfile.File(url2)
h5_file2 = h5py.File(remote_file2)
io2 = pynwb.NWBHDF5IO(file=h5_file2)
nwb2 = io2.read()

# Compare basic session information
print("\n=== Basic Session Information ===")
print(f"File 1 - Subject: {nwb1.subject.subject_id}, Date: {nwb1.session_start_time}")
print(f"File 2 - Subject: {nwb2.subject.subject_id}, Date: {nwb2.session_start_time}")

# Compare trial information
print("\n=== Trial Information ===")
trials1 = nwb1.trials.to_dataframe()
trials2 = nwb2.trials.to_dataframe()
print(f"File 1 - Number of trials: {len(trials1)}")
print(f"File 2 - Number of trials: {len(trials2)}")

# Calculate trial durations
trials1['duration'] = trials1['stop_time'] - trials1['start_time']
trials2['duration'] = trials2['stop_time'] - trials2['start_time']

print("\nTrial duration statistics (seconds):")
print(f"File 1 - Mean: {trials1['duration'].mean():.2f}, Min: {trials1['duration'].min():.2f}, Max: {trials1['duration'].max():.2f}")
print(f"File 2 - Mean: {trials2['duration'].mean():.2f}, Min: {trials2['duration'].min():.2f}, Max: {trials2['duration'].max():.2f}")

# Compare distributions of trial durations
plt.figure(figsize=(12, 5))
plt.hist(trials1['duration'], bins=30, alpha=0.5, label=f'MS13B ({len(trials1)} trials)')
plt.hist(trials2['duration'], bins=30, alpha=0.5, label=f'MS14A ({len(trials2)} trials)')
plt.xlabel('Trial Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Comparison of Trial Durations Between Sessions')
plt.legend()
plt.savefig('explore/trial_duration_comparison.png')

# Compare time series data properties
print("\n=== Time Series Data Properties ===")
ts1 = nwb1.acquisition['time_series']
ts2 = nwb2.acquisition['time_series']
print(f"File 1 - Shape: {ts1.data.shape}, Duration: {ts1.data.shape[0]/ts1.rate:.2f} seconds")
print(f"File 2 - Shape: {ts2.data.shape}, Duration: {ts2.data.shape[0]/ts2.rate:.2f} seconds")

# Compare units information
print("\n=== Units Information ===")
units1 = nwb1.units.to_dataframe()
units2 = nwb2.units.to_dataframe()
print(f"File 1 - Number of units: {len(units1)}")
print(f"File 2 - Number of units: {len(units2)}")

# Compare electrode group locations
print("\n=== Electrode Group Locations ===")
print("File 1:")
for group_name, group in nwb1.electrode_groups.items():
    print(f"  {group_name}: {group.location}")
print("File 2:")
for group_name, group in nwb2.electrode_groups.items():
    print(f"  {group_name}: {group.location}")

# Close the file handles
h5_file1.close()
remote_file1.close()
h5_file2.close()
remote_file2.close()