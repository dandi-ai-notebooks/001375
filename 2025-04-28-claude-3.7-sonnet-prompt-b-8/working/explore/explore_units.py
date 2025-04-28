# This script explores the neural units data from the NWB file
# to understand spike timing patterns and firing rates

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

# Get units data
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")

# Calculate firing rates for each unit
firing_rates = []
unit_ids = []

start_time = 0
end_time = 4800  # Approximately 80 minutes in seconds (based on previous metadata)

# Get trials data for context
trials_df = nwb.trials.to_dataframe()
session_duration = trials_df['stop_time'].max()  # Using last trial's end as session duration
print(f"Session duration: {session_duration:.2f} seconds")

for i, unit_id in enumerate(nwb.units.id[:]):
    spike_times = np.array(nwb.units['spike_times'][i])
    
    # Calculate overall firing rate
    rate = len(spike_times) / (end_time - start_time)
    
    firing_rates.append(rate)
    unit_ids.append(unit_id)
    
    if i < 5:  # Print details for the first 5 units
        print(f"\nUnit ID: {unit_id}")
        print(f"Total spikes: {len(spike_times)}")
        print(f"Firing rate: {rate:.2f} Hz")
        if len(spike_times) > 0:
            print(f"First few spike times: {spike_times[:5]}")
            print(f"Min spike time: {min(spike_times):.2f}, Max spike time: {max(spike_times):.2f}")

# Plot firing rate distribution
plt.figure(figsize=(10, 6))
plt.bar(range(len(firing_rates)), firing_rates)
plt.xlabel('Unit Index')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rates Across Units')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rates.png', dpi=300)

# Plot firing rate histogram
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=15)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Firing Rates')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/firing_rates_histogram.png', dpi=300)

# Plot spike raster for first few units
plt.figure(figsize=(15, 8))
for i in range(min(10, len(nwb.units.id[:]))):
    spike_times = np.array(nwb.units['spike_times'][i])
    
    # Limit to first 5 minutes (300 seconds) for visibility
    mask = spike_times < 300
    spike_times_subset = spike_times[mask]
    
    # Plot spike raster
    plt.plot(spike_times_subset, np.ones_like(spike_times_subset) * i, '|', markersize=4)

plt.xlabel('Time (s)')
plt.ylabel('Unit Index')
plt.title('Spike Raster Plot (First 5 Minutes)')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('explore/spike_raster.png', dpi=300)

# Plot ISI (Inter-Spike Interval) histogram for a sample unit
if len(nwb.units.id[:]) > 0:
    unit_idx = 0  # First unit
    spike_times = np.array(nwb.units['spike_times'][unit_idx])
    
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        
        plt.figure(figsize=(10, 6))
        plt.hist(isis, bins=50)
        plt.xlabel('Inter-Spike Interval (s)')
        plt.ylabel('Count')
        plt.title(f'ISI Histogram for Unit {nwb.units.id[unit_idx]}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('explore/isi_histogram.png', dpi=300)

# Close the file
h5_file.close()

print("\nPlots saved to explore/ directory.")