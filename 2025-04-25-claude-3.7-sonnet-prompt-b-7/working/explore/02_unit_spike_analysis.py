"""
This script explores the spike times data for units (neurons) in the first NWB file.
It creates visualizations of spike times to better understand neural activity patterns.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Set the plotting style for consistent visualizations
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Function to get spike times for a specific unit
def get_spike_times(units, unit_id):
    return units.get_unit_spike_times(unit_id)

# Print basic information about units
units = nwb.units
print(f"Total number of units: {len(units.id)}")

# Create dataframe with unit IDs and spike count
unit_ids = units.id[:]
unit_info = []

print(f"Unit IDs range: {min(unit_ids)} to {max(unit_ids)}")

# Use indices instead of IDs for accessing spike times
for i, unit_id in enumerate(unit_ids):
    spike_times = get_spike_times(units, i)  # Use index i instead of unit_id
    unit_info.append({
        'unit_id': unit_id,
        'spike_count': len(spike_times)
    })

unit_df = pd.DataFrame(unit_info)
unit_df.sort_values('spike_count', ascending=False, inplace=True)

print("\nUnits with most spikes:")
print(unit_df.head(5).to_string(index=False))

print("\nUnits with fewest spikes:")
print(unit_df.tail(5).to_string(index=False))

# Let's analyze the top 5 most active units
top_units = unit_df.head(5)['unit_id'].values
top_unit_indices = [i for i, uid in enumerate(unit_ids) if uid in top_units]
print(f"\nAnalyzing top {len(top_units)} most active units: {top_units}")
print(f"Corresponding indices: {top_unit_indices}")

# Create a directory for images if it doesn't exist
if not os.path.exists('explore/images'):
    os.makedirs('explore/images')

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"\nNumber of trials: {len(trials_df)}")

# 1. Plot spike raster for the top 5 most active units
plt.figure(figsize=(14, 10))
for i, unit_idx in enumerate(top_unit_indices):
    unit_id = unit_ids[unit_idx]
    spike_times = get_spike_times(units, unit_idx)
    
    # Limit to the first 200 seconds for better visualization
    mask = spike_times < 200
    spike_times_subset = spike_times[mask]
    
    plt.plot(spike_times_subset, np.ones_like(spike_times_subset) * (i + 1), '|', 
             markersize=4, label=f"Unit {unit_id}")

plt.xlabel('Time (seconds)')
plt.ylabel('Unit Number')
plt.yticks(range(1, len(top_units) + 1), [f"Unit {id}" for id in top_units])
plt.title('Spike Raster Plot for Top 5 Most Active Units (First 200 seconds)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('explore/images/spike_raster_top_units.png')

# 2. Plot spike histogram (firing rate over time) for the most active unit
most_active_unit = unit_df.iloc[0]['unit_id']
most_active_unit_idx = list(unit_ids).index(most_active_unit)
spike_times = get_spike_times(units, most_active_unit_idx)

plt.figure(figsize=(14, 6))
bin_size = 1  # 1-second bins
max_time = 300  # first 5 minutes
bins = np.arange(0, max_time + bin_size, bin_size)
plt.hist(spike_times[spike_times < max_time], bins=bins, alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Spike Count (per second)')
plt.title(f'Spike Histogram for Unit {most_active_unit} over First 5 Minutes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('explore/images/spike_histogram_most_active.png')

# 3. Calculate and plot average firing rates for all units
firing_rates = []
for i, unit_id in enumerate(unit_ids):
    spike_times = get_spike_times(units, i)
    # Calculate firing rate (spikes per second) over the entire recording
    recording_duration = nwb.acquisition["time_series"].data.shape[0] / nwb.acquisition["time_series"].rate
    rate = len(spike_times) / recording_duration
    firing_rates.append({
        'unit_id': unit_id,
        'firing_rate': rate
    })

firing_rate_df = pd.DataFrame(firing_rates)
firing_rate_df.sort_values('firing_rate', ascending=False, inplace=True)

plt.figure(figsize=(14, 6))
plt.bar(range(len(firing_rate_df)), firing_rate_df['firing_rate'])
plt.xlabel('Unit Rank')
plt.ylabel('Firing Rate (Hz)')
plt.title('Average Firing Rates for All Units')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('explore/images/firing_rates_all_units.png')

print(f"\nAverage firing rate across all units: {firing_rate_df['firing_rate'].mean():.2f} Hz")
print(f"Median firing rate: {firing_rate_df['firing_rate'].median():.2f} Hz")
print(f"Max firing rate: {firing_rate_df['firing_rate'].max():.2f} Hz (Unit {firing_rate_df.iloc[0]['unit_id']})")
print(f"Min firing rate: {firing_rate_df['firing_rate'].min():.2f} Hz (Unit {firing_rate_df.iloc[-1]['unit_id']})")

# 4. Plot spike times relative to trial starts for a single unit
unit_to_analyze = most_active_unit
spike_times = get_spike_times(units, most_active_unit_idx)

# Get trial start and stop times
trial_starts = trials_df['start_time'].values
trial_stops = trials_df['stop_time'].values

# Let's focus on the first 50 trials
num_trials_to_plot = min(50, len(trial_starts))
trial_starts = trial_starts[:num_trials_to_plot]
trial_stops = trial_stops[:num_trials_to_plot]

plt.figure(figsize=(14, 8))

# Plot vertical lines for trial starts
for i, (start, stop) in enumerate(zip(trial_starts, trial_stops)):
    plt.axvspan(start, stop, alpha=0.1, color='gray')
    
    # Find spikes in this trial
    trial_spikes = spike_times[(spike_times >= start) & (spike_times <= stop)]
    
    # Plot spikes for this trial
    plt.plot(trial_spikes, np.ones_like(trial_spikes) * i, '|', markersize=5, color='blue')

plt.xlabel('Time (seconds)')
plt.ylabel('Trial Number')
plt.title(f'Spike Times for Unit {unit_to_analyze} Relative to Trial Starts (First {num_trials_to_plot} Trials)')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.yticks(range(num_trials_to_plot))
plt.savefig('explore/images/spike_times_trial_aligned.png')

# 5. Analyze inter-spike intervals for the most active unit
isi = np.diff(spike_times)

plt.figure(figsize=(14, 6))
plt.hist(isi[isi < 0.5], bins=100, alpha=0.7)
plt.xlabel('Inter-spike Interval (seconds)')
plt.ylabel('Count')
plt.title(f'Inter-spike Interval Distribution for Unit {unit_to_analyze} (intervals < 0.5s)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('explore/images/inter_spike_intervals.png')

# Clean up
io.close()
print("\nScript completed successfully.")