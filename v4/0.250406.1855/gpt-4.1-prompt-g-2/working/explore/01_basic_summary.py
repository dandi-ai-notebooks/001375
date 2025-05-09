# This script loads the NWB file remotely and summarizes key tables:
# - Session and subject metadata
# - Acquisition time series info (shape, dtype, units)
# - Electrodes table columns and counts
# - Trials table columns and number of trials
# - Units table columns and counts

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

print("# Session Info")
print("Identifier:", nwb.identifier)
print("Session description:", nwb.session_description)
print("Session start:", nwb.session_start_time)
print("File create date:", nwb.file_create_date)
subj = nwb.subject
if subj:
    print("Subject ID:", getattr(subj, 'subject_id', 'N/A'))
    print("Subject species:", getattr(subj, 'species', 'N/A'))
    print("Subject sex:", getattr(subj, 'sex', 'N/A'))
    print("Subject age:", getattr(subj, 'age', 'N/A'))
    print("Subject description:", getattr(subj, 'description', 'N/A'))

print("\\n# Acquisition TimeSeries")
ts = nwb.acquisition.get("time_series")
print("  shape:", ts.data.shape, "dtype:", ts.data.dtype)
print("  starting_time:", ts.starting_time, "rate:", ts.rate, "unit:", ts.unit)
print("  First sample:", ts.data[0, :5])

print("\\n# Electrodes Table")
df_elec = nwb.electrodes.to_dataframe()
print("  Columns:", df_elec.columns.tolist())
print("  N electrodes:", df_elec.shape[0])
print("  Example rows:\\n", df_elec.head())

print("\\n# Trials Table")
df_trials = nwb.trials.to_dataframe()
print("  Columns:", df_trials.columns.tolist())
print("  N trials:", df_trials.shape[0])
print("  Example rows:\\n", df_trials.head())

print("\\n# Units Table")
df_units = nwb.units.to_dataframe()
print("  Columns:", df_units.columns.tolist())
print("  N units:", df_units.shape[0])
print("  Example rows:\\n", df_units.head())

io.close()