# This script explores the basic metadata of the NWB file
# to understand the dataset structure and available information

import pynwb
import h5py
import remfile
import pandas as pd

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print("Loading NWB file from:", url)
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata
print("\nBASIC METADATA:")
print(f"Identifier: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")

# Subject information
print("\nSUBJECT INFORMATION:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Device information
print("\nDEVICE INFORMATION:")
for device_name, device in nwb.devices.items():
    print(f"Device name: {device_name}")
    print(f"  Description: {device.description}")
    print(f"  Manufacturer: {device.manufacturer}")

# Electrode groups
print("\nELECTRODE GROUPS:")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group name: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")

# Trials information
print("\nTRIALS INFORMATION:")
print(f"Number of trials: {len(nwb.trials.id)}")
trials_df = nwb.trials.to_dataframe()
print("First 5 trials:")
print(trials_df.head())

# Units information
print("\nUNITS INFORMATION:")
print(f"Number of units: {len(nwb.units.id)}")
print(f"Unit columns: {nwb.units.colnames}")

# Raw data information
print("\nRAW DATA INFORMATION:")
for ts_name, ts in nwb.acquisition.items():
    print(f"Time series name: {ts_name}")
    print(f"  Data shape: {ts.data.shape}")
    print(f"  Data type: {ts.data.dtype}")
    print(f"  Sampling rate: {ts.rate} Hz")
    print(f"  Unit: {ts.unit}")

# Electrodes information
print("\nELECTRODES INFORMATION:")
print(f"Number of electrodes: {len(nwb.electrodes.id)}")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Electrode columns: {list(electrodes_df.columns)}")
print("First 5 electrodes:")
print(electrodes_df.head())