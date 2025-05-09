# This script loads a small time segment (first 1 s) of voltage traces for the first 4 channels
# from the main acquisition time series and plots them to a PNG file.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Extract the main continuous raw ephys, subset in time and channel
ts = nwb.acquisition.get("time_series")
rate = ts.rate  # 30000 Hz
n_channels = 4
duration_s = 1.0
n_samples = int(rate * duration_s)
# Data shape (144M, 256)
data = ts.data[:n_samples, :n_channels] * ts.conversion  # shape (n_samples, n_channels)

plt.figure(figsize=(10, 5))
offset = 0
spacing = 600  # adjust if needed for vertical separation
for ch in range(n_channels):
    plt.plot(np.arange(n_samples) / rate, data[:, ch] + ch * spacing, label=f'Ch {ch+1}')
plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset (mV)')
plt.title('Extracellular traces: first 4 channels, first 1s')
plt.legend()
plt.tight_layout()
plt.savefig('explore/extracellular_traces_4chan_1s.png')
plt.close()

io.close()