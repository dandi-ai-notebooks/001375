import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract spike times for units 20, 21, and 22
units = nwb.units
num_units = min(3, len(units))  # Use a maximum of 3 units
spike_times = []
unit_ids = []
start_index = 20
for i in range(start_index, min(start_index + num_units, len(units))):
    unit_id = units.id[i]
    unit_ids.append(unit_id)
    st = units.spike_times[i]
    spike_times.append(st[(st >= 0) & (st < 10)])

# Plot the spike times
plt.figure(figsize=(10, 6))
for i, st in enumerate(spike_times):
    plt.vlines(st, i, i + 0.8, label=f"Unit {unit_ids[i]}")  # Use unit IDs instead of indices
plt.xlabel("Time (s)")
plt.ylabel("Unit ID")  # Use unit IDs instead of indices
plt.title("Spike times for the first few units (first 10 seconds)")
plt.legend()
plt.savefig("spiketimes.png")
plt.close()