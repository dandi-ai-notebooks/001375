# %%
# This script loads trial start and stop times from an NWB file and plots them.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial start and stop times
trial_start_times = nwb.trials.start_time[:]
trial_stop_times = nwb.trials.stop_time[:]

# Create the plot
plt.figure(figsize=(8, 6))
plt.eventplot(trial_start_times, linelengths=0.5, colors='blue', label='Start Time')
plt.eventplot(trial_stop_times, linelengths=0.5, colors='red', label='Stop Time')
plt.xlabel("Time (s)")
plt.ylabel("Trials")
plt.title("Trial Start and Stop Times")
plt.legend()
plt.savefig("explore/trial_times.png")
plt.close()