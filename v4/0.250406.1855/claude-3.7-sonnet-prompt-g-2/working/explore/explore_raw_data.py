# This script explores the raw electrophysiology data in the NWB file
# and visualizes a small segment of data from a sample of electrodes

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get information about the raw data
time_series = nwb.acquisition["time_series"]
print(f"Data dimensions: {time_series.data.shape}")
print(f"Sampling rate: {time_series.rate} Hz")
print(f"Units: {time_series.unit}")

# Select a small time window to analyze (1 second of data starting at 10 seconds)
start_time = 10  # seconds
duration = 1.0   # seconds
start_idx = int(start_time * time_series.rate)
end_idx = int((start_time + duration) * time_series.rate)

# Select a subset of electrodes (5 from each shank)
electrode_df = nwb.electrodes.to_dataframe()
shank1_electrodes = electrode_df[electrode_df['group_name'] == 'shank1'].head(5).index
shank2_electrodes = electrode_df[electrode_df['group_name'] == 'shank2'].head(5).index

selected_electrodes = list(shank1_electrodes) + list(shank2_electrodes)
selected_labels = [electrode_df.loc[idx, 'label'] for idx in selected_electrodes]

print(f"Selected time window: {start_time}s to {start_time + duration}s")
print(f"Selected electrodes: {selected_labels}")

# Extract the data for the selected time window and electrodes
data_segment = time_series.data[start_idx:end_idx, selected_electrodes]

# Plot the raw data for the selected electrodes
plt.figure(figsize=(12, 10))
t = np.linspace(start_time, start_time + duration, end_idx - start_idx)

for i, (electrode_idx, label) in enumerate(zip(selected_electrodes, selected_labels)):
    # Offset each trace for visibility
    offset = i * 200
    
    # Plot with appropriate offset
    plt.plot(t, data_segment[:, i] + offset, linewidth=0.8, 
             label=f"{label} (#{electrode_idx})")

plt.xlabel('Time (s)')
plt.ylabel('Signal (μV) + Offset')
plt.title('Raw Electrophysiology Data for Selected Electrodes')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('explore/raw_electrophysiology_data.png', dpi=300)
plt.close()

# Plot power spectrum for each electrode
plt.figure(figsize=(12, 8))

for i, (electrode_idx, label) in enumerate(zip(selected_electrodes, selected_labels)):
    signal = data_segment[:, i]
    
    # Compute power spectrum
    freq = np.fft.fftfreq(len(signal), 1/time_series.rate)
    ps = np.abs(np.fft.fft(signal))**2
    
    # Plot only positive frequencies up to 500Hz for clarity
    mask = (freq > 0) & (freq <= 500)
    plt.semilogy(freq[mask], ps[mask], linewidth=0.8, alpha=0.7,
                 label=f"{label} (#{electrode_idx})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum (0-500 Hz)')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('explore/power_spectrum.png', dpi=300)
plt.close()

# Create a pairwise correlation matrix for the 10 selected electrodes
plt.figure(figsize=(10, 8))

# Calculate correlation matrix
corr_matrix = np.corrcoef(data_segment.T)

# Plot as heatmap
plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(np.arange(len(selected_labels)), selected_labels, rotation=90, fontsize=8)
plt.yticks(np.arange(len(selected_labels)), selected_labels, fontsize=8)
plt.title('Signal Correlation Between Electrodes')
plt.tight_layout()
plt.savefig('explore/electrode_correlation.png', dpi=300)
plt.close()

print("Analysis completed and plots saved to 'explore' directory")