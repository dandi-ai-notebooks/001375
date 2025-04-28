# This script explores the basic metadata of the NWB file
# to understand the structure and content of the data

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("\n===== Basic Metadata =====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# Print subject information
print("\n===== Subject Information =====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Description: {nwb.subject.description}")

# Print electrode group information
print("\n===== Electrode Groups =====")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device description: {group.device.description}")
    print(f"  Device manufacturer: {group.device.manufacturer}")

# Print trial information
print("\n===== Trials Information =====")
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print("First 5 trials:")
print(trials_df.head())
print("\nDescriptive statistics of trial durations (seconds):")
durations = trials_df['stop_time'] - trials_df['start_time']
print(durations.describe())

# Print unit information
print("\n===== Units Information =====")
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
print("First 5 units:")
print(units_df.head())

# Print electrode information
print("\n===== Electrodes Information =====")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")
print("First 5 electrodes:")
print(electrodes_df.head())

# Get group counts for electrodes
print("\nElectrode group distribution:")
group_counts = electrodes_df['group_name'].value_counts()
for group, count in group_counts.items():
    print(f"  {group}: {count} electrodes")

# Get information about time series data
print("\n===== Time Series Information =====")
time_series = nwb.acquisition["time_series"]
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Unit: {time_series.unit}")
print(f"Data shape: {time_series.data.shape}")
print(f"Data type: {time_series.data.dtype}")

# Calculate recording duration
duration_samples = time_series.data.shape[0]
duration_seconds = duration_samples / time_series.rate
duration_minutes = duration_seconds / 60
print(f"Recording duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")

# Close the file
h5_file.close()