# Explore main metadata and preview signals in NWB file
# Output: print summary and save a preview plot (mean raw trace from several channels for 1 min)
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print top-level metadata
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject: {nwb.subject.subject_id}, species: {nwb.subject.species}, sex: {nwb.subject.sex}, age: {nwb.subject.age}")
print(f"Subject description: {nwb.subject.description}")
print(f"Trials: {nwb.trials.to_dataframe().shape[0]}")
print(f"Units: {nwb.units.to_dataframe().shape[0]}")
print(f"Electrodes: {nwb.electrodes.to_dataframe().shape[0]}")
print(f"Main acquisition data key: {list(nwb.acquisition.keys())}")

# Show shape of extracellular data
data = nwb.acquisition["time_series"].data
print(f"Extracellular data shape: {data.shape}; dtype: {data.dtype}")
print(f"Channel count: {data.shape[1]}; Sample count: {data.shape[0]}; Sample rate: {nwb.acquisition['time_series'].rate} Hz")
duration_sec = data.shape[0] / nwb.acquisition["time_series"].rate
print(f"Total duration (s): {duration_sec:.1f}")

# Preview a mean trace for first minute from several channels (check for existence and valid data)
preview_duration_sec = 60  # Only first 60 seconds
preview_samples = int(preview_duration_sec * nwb.acquisition["time_series"].rate)
n_channels_to_plot = 4
plt.figure(figsize=(10,6))
t = np.arange(preview_samples) / nwb.acquisition["time_series"].rate
for i in range(n_channels_to_plot):
    if i < data.shape[1]:
        channel_data = data[0:preview_samples, i]
        plt.plot(t, channel_data, label=f"Channel {i}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (int16, raw units)")
plt.title("Extracellular traces, first 60 s, first 4 channels")
plt.legend()
plt.tight_layout()
plt.savefig("explore/preview_traces.png")
print("Saved preview plot to explore/preview_traces.png")

io.close()
h5_file.close()
remote_file.close()