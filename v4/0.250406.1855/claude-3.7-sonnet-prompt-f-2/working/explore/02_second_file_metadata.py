# This script explores the basic metadata of the second NWB file
# to compare with the first file and understand differences

import pynwb
import h5py
import remfile
import pandas as pd

# Load the second NWB file
url = "https://api.dandiarchive.org/api/assets/376dc673-611b-4521-b135-9ec01c7f4f74/download/"
print("Loading second NWB file from:", url)
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