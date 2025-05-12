# This script explores the basic structure and metadata of the first NWB file
# to understand the dataset organization and available data.

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print("\n--- Basic NWB File Information ---")
print(f"Identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File create date: {nwb.file_create_date[0]}")

# Subject information
print("\n--- Subject Information ---")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Electrode information
print("\n--- Electrode Information ---")
print(f"Number of electrodes: {len(nwb.electrodes)}")
print(f"Electrode columns: {nwb.electrodes.colnames}")

# Show first 5 electrodes
print("\n--- First 5 Electrodes ---")
electrode_df = nwb.electrodes.to_dataframe()
print(electrode_df.head())

# Electrode group information
print("\n--- Electrode Groups ---")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device.description} (Manufacturer: {group.device.manufacturer})")

# Time series information
print("\n--- Time Series Information ---")
time_series = nwb.acquisition["time_series"]
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Unit: {time_series.unit}")
print(f"Data shape: {time_series.data.shape}")
print(f"Data type: {time_series.data.dtype}")
print(f"Starting time: {time_series.starting_time} {time_series.starting_time_unit}")
print(f"Conversion: {time_series.conversion}")
print(f"Offset: {time_series.offset}")

# Trials information
print("\n--- Trials Information ---")
print(f"Number of trials: {len(nwb.trials)}")
print(f"Trial columns: {nwb.trials.colnames}")

# Show first 5 trials
print("\n--- First 5 Trials ---")
trials_df = nwb.trials.to_dataframe()
print(trials_df.head())

# Units information
print("\n--- Units Information ---")
print(f"Number of units: {len(nwb.units)}")
print(f"Unit columns: {nwb.units.colnames}")
print(f"Waveform unit: {nwb.units.waveform_unit}")

# Show spike counts for first 5 units
print("\n--- First 5 Units Spike Counts ---")
for i in range(min(5, len(nwb.units))):
    spike_times = nwb.units["spike_times"][i]
    print(f"Unit {i}: {len(spike_times)} spikes")