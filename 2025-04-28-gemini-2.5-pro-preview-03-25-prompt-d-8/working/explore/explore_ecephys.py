# Explore raw ephys data (TimeSeries)
# Load a short segment (0.1 seconds) of data for the first 4 channels.
# Plot the time series traces for these channels.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print('Loading NWB file...')
# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print('Accessing TimeSeries data...')
ts = nwb.acquisition['time_series']
sampling_rate = ts.rate
num_channels_to_plot = 4
duration_to_plot = 0.1 # seconds
start_index = 0 # Start from the beginning
end_index = int(duration_to_plot * sampling_rate)

print(f'Loading data for the first {num_channels_to_plot} channels for {duration_to_plot} seconds...')
# Note: Loading data chunk by chunk might be necessary for larger requests in a real scenario
data_segment = ts.data[start_index:end_index, :num_channels_to_plot]
time_vector = np.linspace(start_index / sampling_rate, end_index / sampling_rate, end_index - start_index)

print('Plotting time series traces...')
# Plotting
sns.set_theme() # Use seaborn style
plt.figure(figsize=(12, 6))
for i in range(num_channels_to_plot):
    # Offset traces for visibility
    plt.plot(time_vector, data_segment[:, i] + i * np.mean(np.abs(data_segment[:, i])) * 3, label=f'Channel {i+1}')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV, offset)')
plt.title(f'Raw Ephys Traces (First {duration_to_plot}s, Channels 1-{num_channels_to_plot})')
plt.legend(loc='upper right')
plt.yticks([]) # Hide y-axis ticks as traces are offset
plt.savefig('explore/ecephys_traces.png')
plt.close() # Close the plot to prevent hanging

print('Plot saved to explore/ecephys_traces.png')
print('Script finished.')