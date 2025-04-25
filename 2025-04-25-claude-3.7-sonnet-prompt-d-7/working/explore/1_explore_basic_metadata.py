# Script to explore basic metadata and structure of the NWB file from sub-MS13B

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

# Print basic metadata
print("\n--- Basic Metadata ---")
print(f"NWB Identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject description: {nwb.subject.description}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")

# Display information about electrode groups
print("\n--- Electrode Groups ---")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device description: {group.device.description}")
    print(f"  Device manufacturer: {group.device.manufacturer}")

# Get info about trials/laps
print("\n--- Trials/Laps Information ---")
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print(f"Trial duration statistics (seconds):")
trial_durations = trials_df['stop_time'] - trials_df['start_time']
print(f"  Min: {trial_durations.min():.2f}")
print(f"  Max: {trial_durations.max():.2f}")
print(f"  Mean: {trial_durations.mean():.2f}")
print(f"  Median: {trial_durations.median():.2f}")
print(f"First 5 trials:")
print(trials_df.head())

# Plot trial durations
plt.figure(figsize=(10, 5))
plt.hist(trial_durations, bins=30)
plt.title('Trial/Lap Duration Distribution')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig('explore/trial_durations.png')

# Get info about units (sorted neurons)
print("\n--- Units (Sorted Neurons) Information ---")
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")

# Print spike counts for each unit
spike_counts = []
for i in range(len(units_df)):
    spike_times = nwb.units['spike_times'][i]
    spike_counts.append(len(spike_times))
    
print(f"Spike count statistics:")
print(f"  Min spikes: {min(spike_counts)}")
print(f"  Max spikes: {max(spike_counts)}")
print(f"  Mean spikes: {np.mean(spike_counts):.2f}")
print(f"  Median spikes: {np.median(spike_counts):.2f}")

# Plot spike counts
plt.figure(figsize=(10, 5))
plt.hist(spike_counts, bins=15)
plt.title('Spike Count Distribution Across Units')
plt.xlabel('Number of Spikes')
plt.ylabel('Number of Units')
plt.savefig('explore/spike_counts.png')

# Get info about raw data dimensions
print("\n--- Raw Data Information ---")
time_series = nwb.acquisition["time_series"]
print(f"Data shape: {time_series.data.shape}")
print(f"Data type: {time_series.data.dtype}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Units: {time_series.unit}")

# Close the file
io.close()
h5_file.close()
remote_file.close()