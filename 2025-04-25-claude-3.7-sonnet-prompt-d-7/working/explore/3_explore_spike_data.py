# Script to explore unit spike trains and patterns

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

# Get basic units information
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")

# Select a few units to analyze (1st, middle, and last units)
unit_indices = [0, len(units_df)//2, len(units_df)-1]
print(f"Analyzing units at indices: {unit_indices}")

# Get trial information to align spikes
trials_df = nwb.trials.to_dataframe()
trial_start_times = trials_df['start_time'].values
trial_end_times = trials_df['stop_time'].values

# Plotting spike rasters and firing rates
plt.figure(figsize=(15, 10))

for i, unit_idx in enumerate(unit_indices):
    # Get spike times for the unit
    spike_times = nwb.units['spike_times'][unit_idx]
    print(f"Unit {unit_idx} has {len(spike_times)} spikes")
    
    # Plot spike raster (first 50 trials for clarity)
    plt.subplot(3, 2, 2*i + 1)
    for j, (start, end) in enumerate(zip(trial_start_times[:50], trial_end_times[:50])):
        # Get spikes within this trial
        trial_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
        # Normalize to trial start
        normalized_spikes = trial_spikes - start
        # Plot spike raster
        if len(normalized_spikes) > 0:
            plt.plot(normalized_spikes, np.ones_like(normalized_spikes) * j, '|', markersize=4)
    
    plt.title(f'Unit {unit_idx} Spike Raster (First 50 Trials)')
    plt.xlabel('Time from trial start (s)')
    plt.ylabel('Trial #')
    
    # Calculate and plot firing rate over time (bin spikes in 60 second bins)
    plt.subplot(3, 2, 2*i + 2)
    
    # Define the time range for the entire recording
    first_spike = min(spike_times)
    last_spike = max(spike_times)
    
    # Create time bins (60 second bins)
    bin_size = 60  # seconds
    bins = np.arange(0, last_spike + bin_size, bin_size)
    
    # Count spikes in each bin
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    
    # Calculate firing rates (spikes/second)
    firing_rates = spike_counts / bin_size
    
    # Plot firing rate over time
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers/60, firing_rates)  # Convert x-axis to minutes
    
    plt.title(f'Unit {unit_idx} Firing Rate Over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Firing Rate (Hz)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('explore/spike_patterns.png')

# Create inter-spike interval (ISI) histograms for each unit
plt.figure(figsize=(15, 5))
for i, unit_idx in enumerate(unit_indices):
    spike_times = nwb.units['spike_times'][unit_idx]
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    # Plot ISI distribution
    plt.subplot(1, 3, i + 1)
    plt.hist(isis, bins=50, range=(0, 0.5))  # ISIs up to 500 ms
    plt.title(f'Unit {unit_idx} ISI Distribution')
    plt.xlabel('Inter-Spike Interval (s)')
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('explore/isi_distributions.png')

# Close the file
io.close()
h5_file.close()
remote_file.close()