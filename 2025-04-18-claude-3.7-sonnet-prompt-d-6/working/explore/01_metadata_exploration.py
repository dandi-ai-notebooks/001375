#!/usr/bin/env python
"""
This script explores the basic metadata and structure of one NWB file from Dandiset 001375.
The goal is to understand what data is available and how it is organized.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print("\n--- Basic Information ---")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")

# Print subject information
print("\n--- Subject Information ---")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Description: {nwb.subject.description}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# Show electrode group information
print("\n--- Electrode Groups ---")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description} ({group.device.manufacturer})")

# Look at the first few trials
print("\n--- Trials ---")
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print("First 5 trials:")
print(trials_df.head())

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
print("\nTrial duration statistics (seconds):")
print(f"  Mean: {trials_df['duration'].mean():.2f}")
print(f"  Min: {trials_df['duration'].min():.2f}")
print(f"  Max: {trials_df['duration'].max():.2f}")
print(f"  Median: {trials_df['duration'].median():.2f}")

# Plot trial durations
plt.figure(figsize=(10, 5))
plt.hist(trials_df['duration'], bins=30)
plt.title('Trial Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.savefig('explore/trial_durations.png')

# Look at units information
print("\n--- Units ---")
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
print("First 5 units:")
print(units_df.head())

# Look at the electrodes table
print("\n--- Electrodes ---")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")
print("First 5 electrodes:")
print(electrodes_df.head())

# Examine the time series data shape
print("\n--- Time Series Data ---")
time_series = nwb.acquisition['time_series']
print(f"Time series shape: {time_series.data.shape}")
print(f"Time series unit: {time_series.unit}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Duration in seconds: {time_series.data.shape[0] / time_series.rate:.2f}")

# Close the file handles
h5_file.close()
remote_file.close()