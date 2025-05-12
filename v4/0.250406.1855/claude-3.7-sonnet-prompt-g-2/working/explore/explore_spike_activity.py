# This script explores spike activity of units in the NWB file
# and creates visualizations of spike times and firing rates

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the first NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session duration: {nwb.trials['stop_time'][-1]} seconds")

# Get trial information
trials_df = nwb.trials.to_dataframe()
print(f"Number of trials: {len(trials_df)}")

# Select 3 units with different spike counts for visualization
selected_units = [0, 1, 2]  # First three units
unit_ids = nwb.units.id.data[:]
print(f"Selected units: {[unit_ids[i] for i in selected_units]}")

# Get spike times for selected units
spike_times_list = []
for unit_idx in selected_units:
    spike_times = nwb.units["spike_times"][unit_idx]
    spike_times_list.append(spike_times)
    print(f"Unit {unit_ids[unit_idx]}: {len(spike_times)} spikes")

# Create a raster plot for selected units during first 100 seconds
plt.figure(figsize=(12, 6))
for i, spike_times in enumerate(spike_times_list):
    # Filter spike times to first 100 seconds
    spikes_in_window = spike_times[spike_times < 100]
    plt.plot(spikes_in_window, np.ones_like(spikes_in_window) * i + 1, '|', markersize=4)

plt.yticks(np.arange(1, len(selected_units) + 1), [f"Unit {unit_ids[i]}" for i in selected_units])
plt.xlabel('Time (s)')
plt.title('Spike Raster Plot (First 100 seconds)')
plt.tight_layout()
plt.savefig('explore/spike_raster_plot.png', dpi=300)
plt.close()

# Create firing rate histograms (bin size: 0.5 seconds)
plt.figure(figsize=(12, 8))
bin_size = 0.5  # seconds
bin_edges = np.arange(0, 100 + bin_size, bin_size)
bin_centers = bin_edges[:-1] + bin_size / 2

for i, (unit_idx, spike_times) in enumerate(zip(selected_units, spike_times_list)):
    # Filter spike times to first 100 seconds
    spikes_in_window = spike_times[spike_times < 100]
    
    # Compute histogram
    spike_hist, _ = np.histogram(spikes_in_window, bins=bin_edges)
    firing_rate = spike_hist / bin_size  # Convert to Hz
    
    plt.subplot(len(selected_units), 1, i + 1)
    plt.bar(bin_centers, firing_rate, width=bin_size * 0.8, alpha=0.7)
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Unit {unit_ids[unit_idx]} Firing Rate')
    
    # Show trial boundaries
    for _, trial in trials_df.iterrows():
        start, stop = trial['start_time'], trial['stop_time']
        if start < 100:
            plt.axvspan(start, min(stop, 100), alpha=0.2, color='gray')

# Add common labels
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('explore/firing_rate_histograms.png', dpi=300)
plt.close()

# Create a comparison of average firing rates across trials
plt.figure(figsize=(10, 6))

# Get first 20 trials
trial_subset = trials_df.iloc[:20]

# Calculate firing rates for each trial and unit
avg_rates = []

for unit_idx, spike_times in zip(selected_units, spike_times_list):
    unit_rates = []
    
    for _, trial in trial_subset.iterrows():
        start, stop = trial['start_time'], trial['stop_time']
        trial_duration = stop - start
        
        # Count spikes in this trial
        trial_spikes = np.sum((spike_times >= start) & (spike_times < stop))
        rate = trial_spikes / trial_duration
        unit_rates.append(rate)
    
    avg_rates.append(unit_rates)

# Plot bars for average firing rates across trials
trial_ids = np.arange(len(trial_subset))
bar_width = 0.25
positions = [trial_ids, trial_ids + bar_width, trial_ids + 2 * bar_width]

for i, rates in enumerate(avg_rates):
    plt.bar(positions[i], rates, width=bar_width, 
            alpha=0.7, label=f'Unit {unit_ids[selected_units[i]]}')

plt.xlabel('Trial Number')
plt.ylabel('Average Firing Rate (Hz)')
plt.title('Average Firing Rates Across First 20 Trials')
plt.legend()
plt.tight_layout()
plt.savefig('explore/avg_firing_rates_by_trial.png', dpi=300)
plt.close()

print("Plots saved to 'explore' directory")