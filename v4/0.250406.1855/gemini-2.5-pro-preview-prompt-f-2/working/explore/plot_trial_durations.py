# Purpose: Load trial start and stop times, calculate trial durations,
# and plot a histogram of these durations to understand their distribution.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn styling
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # ensure read-only mode
nwb = io.read()

# Access trials table
trials_table = nwb.trials

if trials_table is None:
    print("No 'trials' table found in the NWB file.")
    # Create an empty plot or a message and save it
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "No 'trials' table found.",
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title('Trial Durations')
    plt.tight_layout()
    plt.savefig('explore/trial_durations_hist.png')
    plt.close(fig)
else:
    start_times = trials_table['start_time'][:]
    stop_times = trials_table['stop_time'][:]
    
    if len(start_times) == 0:
        print("Trials table is empty.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Trials table is empty.",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Trial Durations')
    else:
        trial_durations = stop_times - start_times
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(trial_durations, bins=30, edgecolor='black')
        plt.xlabel('Trial Duration (s)')
        plt.ylabel('Number of Trials')
        plt.title('Distribution of Trial Durations')
        mean_duration = np.mean(trial_durations)
        median_duration = np.median(trial_durations)
        plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_duration:.2f}s')
        plt.axvline(median_duration, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_duration:.2f}s')
        plt.legend()

    plt.tight_layout()
    plt.savefig('explore/trial_durations_hist.png')
    plt.close() # Close the figure

io.close()
print("Plot saved to explore/trial_durations_hist.png")