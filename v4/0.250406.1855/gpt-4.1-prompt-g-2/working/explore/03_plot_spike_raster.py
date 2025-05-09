# This script plots a spike raster for the first 5 units in the NWB file, for spike times < 30 s

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

df_units = nwb.units.to_dataframe()
unit_ids = df_units.index[:5]
plt.figure(figsize=(10, 4))
for i, unit_id in enumerate(unit_ids):
    spike_times = df_units.loc[unit_id, 'spike_times']
    # Select spikes within first 30 seconds
    spike_times = [t for t in spike_times if t < 30.0]
    plt.vlines(spike_times, i + 0.5, i + 1.5)
plt.yticks(range(1, 6), [f'Unit {x}' for x in unit_ids])
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title('Spike raster: first 5 units, spike times < 30 s')
plt.tight_layout()
plt.savefig('explore/spike_raster_5units.png')
plt.close()

io.close()