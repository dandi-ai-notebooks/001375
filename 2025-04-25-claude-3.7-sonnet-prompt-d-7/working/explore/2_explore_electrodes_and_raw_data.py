# Script to explore electrodes and raw electrophysiology data

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode information
print("\n--- Electrode Information ---")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Number of electrodes: {len(electrodes_df)}")
print("\nFirst 5 electrodes:")
print(electrodes_df.head())

# Count electrodes by group
group_counts = electrodes_df['group_name'].value_counts()
print("\nElectrodes by group:")
print(group_counts)

# Get sample of raw data (use a small time window to avoid memory issues)
print("\n--- Raw Data Sample ---")
time_series = nwb.acquisition["time_series"]
# Sample 1 second of data from 10 channels (first 10 channels for simplicity)
sample_length = 30000  # 1 second at 30kHz
sample_channels = 10
raw_data_sample = time_series.data[0:sample_length, 0:sample_channels]
print(f"Raw data sample shape: {raw_data_sample.shape}")

# Calculate some basic statistics on the sample
print(f"Min value: {np.min(raw_data_sample)}")
print(f"Max value: {np.max(raw_data_sample)}")
print(f"Mean value: {np.mean(raw_data_sample)}")
print(f"Standard deviation: {np.std(raw_data_sample)}")

# Plot a short segment (100ms) of raw data for a few channels
plt.figure(figsize=(12, 8))
sample_time = np.arange(3000) / time_series.rate  # 100ms at 30kHz
for i in range(min(5, sample_channels)):
    plt.plot(sample_time, raw_data_sample[:3000, i] + i*200, label=f'Channel {i}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV, offset for clarity)')
plt.title('Raw Electrophysiology Data (First 100ms, First 5 Channels)')
plt.legend()
plt.savefig('explore/raw_data_sample.png')

# Plot a sample of frequency content of the raw data
plt.figure(figsize=(10, 6))
# Use a longer window for frequency analysis (1 second)
for i in range(min(3, sample_channels)):
    # Compute FFT
    signal = raw_data_sample[:, i]
    fft_vals = np.absolute(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(len(signal), 1.0/time_series.rate)
    
    # Plot only up to 1000 Hz for visibility
    mask = fft_freq <= 1000
    plt.semilogy(fft_freq[mask], fft_vals[mask], label=f'Channel {i}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (log scale)')
plt.title('Frequency Content of Raw Data (First 3 Channels, up to 1000 Hz)')
plt.legend()
plt.grid(True)
plt.savefig('explore/raw_data_fft.png')

# Close the file
io.close()
h5_file.close()
remote_file.close()