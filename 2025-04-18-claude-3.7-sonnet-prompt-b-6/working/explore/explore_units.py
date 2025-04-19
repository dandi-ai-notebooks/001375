"""
This script explores the neural units (neurons) in the MS13B NWB file.
The goal is to understand the spiking properties of neurons and their relationship to the trials.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get units information
units_df = nwb.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")
print(f"Available columns: {units_df.columns.tolist()}")

# Get trial information
trials_df = nwb.intervals['trials'].to_dataframe()
print(f"\nTotal number of trials: {len(trials_df)}")
print(f"Trial information columns: {trials_df.columns.tolist()}")
print(f"Trial duration stats (seconds):")
trial_durations = trials_df['stop_time'] - trials_df['start_time']
print(trial_durations.describe())

# Get first 5 units spike times
print("\nExtracting spike times for the first 5 units...")
for i in range(min(5, len(units_df))):
    unit_id = units_df.index[i]
    spike_times = nwb.units['spike_times'][i]
    print(f"Unit {unit_id}: {len(spike_times)} spikes")
    print(f"  First 5 spike times: {spike_times[:5]}")
    print(f"  Min spike time: {spike_times.min()}, Max spike time: {spike_times.max()}")
    
    # Calculate firing rate
    recording_duration = trials_df['stop_time'].max()  # Use the end of the last trial as the recording duration
    firing_rate = len(spike_times) / recording_duration
    print(f"  Overall firing rate: {firing_rate:.2f} Hz")

# Plot spike raster for a few units
plt.figure(figsize=(15, 10))
num_units_to_plot = min(10, len(units_df))
for i in range(num_units_to_plot):
    unit_id = units_df.index[i]
    spike_times = nwb.units['spike_times'][i]
    plt.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=4)

plt.xlabel('Time (s)')
plt.ylabel('Unit #')
plt.title('Spike Raster Plot for First 10 Units')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.savefig('explore/spike_raster.png')

# Plot trial boundaries on top of the raster plot
plt.figure(figsize=(15, 10))
num_units_to_plot = min(10, len(units_df))
for i in range(num_units_to_plot):
    unit_id = units_df.index[i]
    spike_times = nwb.units['spike_times'][i]
    plt.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=4)

# Add trial boundaries as vertical lines (only showing first 20 trials to avoid clutter)
for i, trial in trials_df.iloc[:20].iterrows():
    plt.axvline(x=trial['start_time'], color='r', linestyle='--', alpha=0.3)
    
plt.xlabel('Time (s)')
plt.ylabel('Unit #')
plt.title('Spike Raster Plot with Trial Boundaries')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.savefig('explore/spike_raster_with_trials.png')

# Plot firing rate histogram
plt.figure(figsize=(12, 6))
all_firing_rates = []
for i in range(len(units_df)):
    spike_times = nwb.units['spike_times'][i]
    recording_duration = trials_df['stop_time'].max()
    firing_rate = len(spike_times) / recording_duration
    all_firing_rates.append(firing_rate)

plt.hist(all_firing_rates, bins=20, alpha=0.7)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Firing Rates')
plt.grid(True)
plt.savefig('explore/firing_rate_distribution.png')

# Look at spike timing relative to trials
# For each unit, compute PSTH aligned to trial start
trial_starts = trials_df['start_time'].values
window = [-1, 3]  # 1 second before to 3 seconds after trial start
bin_size = 0.1  # 100 ms bins
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Choose a few units for PSTH analysis
units_for_psth = min(5, len(units_df))
psth_data = np.zeros((units_for_psth, len(bins)-1))

plt.figure(figsize=(12, 8))
for i in range(units_for_psth):
    unit_id = units_df.index[i]
    spike_times = nwb.units['spike_times'][i]
    
    # Compute PSTH for this unit
    unit_psth = np.zeros(len(bins)-1)
    for t_start in trial_starts:
        # Align spikes to trial start
        aligned_spikes = spike_times - t_start
        # Count spikes in each bin
        hist, _ = np.histogram(aligned_spikes, bins=bins)
        unit_psth += hist
    
    # Normalize by number of trials and bin size to get firing rate
    unit_psth = unit_psth / (len(trial_starts) * bin_size)
    psth_data[i] = unit_psth
    
    # Plot PSTH for this unit
    plt.plot(bin_centers, unit_psth, label=f'Unit {unit_id}')

plt.axvline(x=0, color='k', linestyle='--')  # Trial start
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Firing rate (Hz)')
plt.title('Peri-Stimulus Time Histogram (PSTH)')
plt.legend()
plt.grid(True)
plt.savefig('explore/psth.png')

# Close the file
h5_file.close()