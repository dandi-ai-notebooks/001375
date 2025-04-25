"""
This script explores the basic metadata of the first NWB file in Dandiset 001375.
It provides an overview of the file structure, subject information, and available data types.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("\n=== Basic Metadata ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# Print subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Print electrode group information
print("\n=== Electrode Groups ===")
for name, group in nwb.electrode_groups.items():
    print(f"Group: {name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description} (Manufacturer: {group.device.manufacturer})")

# Get a count of all electrodes
print("\n=== Electrodes ===")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Total electrode count: {len(electrodes_df)}")

# Print unique electrode locations
unique_locations = electrodes_df['location'].unique()
print("Unique electrode locations:")
for location in unique_locations:
    count = len(electrodes_df[electrodes_df['location'] == location])
    print(f"  {location}: {count} electrodes")

# Print unique electrode group names
unique_groups = electrodes_df['group_name'].unique()
print("\nUnique electrode group names:")
for group in unique_groups:
    count = len(electrodes_df[electrodes_df['group_name'] == group])
    print(f"  {group}: {count} electrodes")

# Print trial information
print("\n=== Trials ===")
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")
print("First 5 trials:")
print(trials_df.head(5))

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
print(f"\nAverage trial duration: {trials_df['duration'].mean():.2f} seconds")
print(f"Min trial duration: {trials_df['duration'].min():.2f} seconds")
print(f"Max trial duration: {trials_df['duration'].max():.2f} seconds")

# Print unit (neuron) information
print("\n=== Units (Neurons) ===")
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")

# Print raw data information
print("\n=== Raw Time Series Data ===")
time_series = nwb.acquisition["time_series"]
print(f"Time series data shape: {time_series.data.shape}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Data unit: {time_series.unit}")
print(f"Duration: {time_series.data.shape[0] / time_series.rate:.2f} seconds")
print(f"Number of channels: {time_series.data.shape[1]}")

# Clean up
io.close()
print("\nScript completed successfully.")