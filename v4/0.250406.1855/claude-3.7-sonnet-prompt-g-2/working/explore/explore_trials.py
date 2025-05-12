# This script explores trial structure and behavioral correlates in the data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"Total number of trials: {len(trials_df)}")

# Calculate trial durations
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
print(f"Mean trial duration: {np.mean(trials_df['duration']):.3f} s")
print(f"Median trial duration: {np.median(trials_df['duration']):.3f} s")
print(f"Min trial duration: {np.min(trials_df['duration']):.3f} s")
print(f"Max trial duration: {np.max(trials_df['duration']):.3f} s")

# Plot trial durations
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(range(len(trials_df)), trials_df['duration'], 'o-', markersize=3)
plt.xlabel('Trial Number')
plt.ylabel('Duration (s)')
plt.title('Trial Durations')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.hist(trials_df['duration'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.title('Trial Duration Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explore/trial_durations.png', dpi=300)
plt.close()

# Select a few units to analyze their activity during trials
selected_units = [0, 1, 2]  # First three units
unit_ids = nwb.units.id.data[:]

# Get spike times for the selected units
spike_times_list = []
for unit_idx in selected_units:
    spike_times = nwb.units["spike_times"][unit_idx]
    spike_times_list.append(spike_times)
    print(f"Unit {unit_ids[unit_idx]}: {len(spike_times)} spikes")

# Calculate firing rates during each trial for selected units
trial_firing_rates = np.zeros((len(trials_df), len(selected_units)))

for t, (_, trial) in enumerate(trials_df.iterrows()):
    start, stop = trial['start_time'], trial['stop_time']
    duration = stop - start
    
    for u, (unit_idx, spike_times) in enumerate(zip(selected_units, spike_times_list)):
        # Count spikes in this trial
        trial_spikes = np.sum((spike_times >= start) & (spike_times < stop))
        rate = trial_spikes / duration
        trial_firing_rates[t, u] = rate

# Plot firing rates across trials
plt.figure(figsize=(12, 10))

for u, unit_idx in enumerate(selected_units):
    plt.subplot(len(selected_units), 1, u+1)
    plt.plot(range(len(trials_df)), trial_firing_rates[:, u], 'o-', markersize=2)
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]} Firing Rate Across Trials')
    plt.grid(True, alpha=0.3)

plt.xlabel('Trial Number')
plt.tight_layout()
plt.savefig('explore/firing_rates_across_trials.png', dpi=300)
plt.close()

# Check for correlation between trial duration and firing rates
plt.figure(figsize=(15, 5))
for u, unit_idx in enumerate(selected_units):
    plt.subplot(1, len(selected_units), u+1)
    
    # Calculate correlation
    r, p = stats.pearsonr(trials_df['duration'], trial_firing_rates[:, u])
    
    plt.scatter(trials_df['duration'], trial_firing_rates[:, u], alpha=0.5)
    plt.xlabel('Trial Duration (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]}: r={r:.3f}, p={p:.3f}')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explore/duration_vs_firing_rate.png', dpi=300)
plt.close()

# Analyze trial-to-trial variability in firing
plt.figure(figsize=(10, 6))
plt.boxplot([trial_firing_rates[:, u] for u in range(len(selected_units))], 
           labels=[f'Unit {unit_ids[i]}' for i in selected_units])
plt.ylabel('Firing Rate (Hz)')
plt.title('Distribution of Firing Rates Across Trials')
plt.grid(True, alpha=0.3)
plt.savefig('explore/firing_rate_distributions.png', dpi=300)
plt.close()

print("Trial analysis completed and plots saved to 'explore' directory")