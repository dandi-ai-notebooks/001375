# Script to explore trial information from the NWB file
# https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/
# It calculates trial durations and saves a histogram plot.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
print("NWB file loaded.")

# Access the trials table
trials = nwb.trials
print(f"Accessed 'trials' table with columns: {trials.colnames}")
print(f"Number of trials: {len(trials.start_time)}")

# Calculate trial durations
start_times = trials.start_time[:]
stop_times = trials.stop_time[:]
durations = stop_times - start_times
print(f"Calculated durations for {len(durations)} trials.")

# Create and save the histogram
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=30, edgecolor='black')
plt.title('Histogram of Trial Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Trials')
plt.grid(axis='y', alpha=0.75)

plot_path = "explore/trial_durations.png"
plt.savefig(plot_path)
print(f"Histogram saved to: {plot_path}")

# Close the file handles
io.close()
print("File handles closed.")