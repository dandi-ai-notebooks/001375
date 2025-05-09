# Purpose: Load spike times for a few units and create a raster plot
# to visualize their firing patterns.

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

units_df = nwb.units.to_dataframe()
num_units_to_plot = min(5, len(units_df)) # Plot up to 5 units, or fewer if not available
unit_ids_to_plot = units_df.index[:num_units_to_plot]

spike_times_list = []
labels = []

print(f"Number of units found: {len(units_df)}")
print(f"Plotting spike times for the first {num_units_to_plot} units.")

for i, unit_id in enumerate(unit_ids_to_plot):
    # Access spike times for the unit.
    # nwb.units['spike_times'] is a RaggedArray; we access its elements.
    # The index into spike_times corresponds to the row index in the units table.
    # We need to find the row index for the current unit_id if they are not sequential from 0.
    # However, units_df.index gives the unit_id, and we can get its positional index
    # or assume that spike_times_index aligns with the positional order of units.
    # Let's get the positional index of the unit_id in the dataframe's index
    unit_row_idx = units_df.index.get_loc(unit_id)
    
    # Access spike times for the unit at unit_row_idx
    st = nwb.units['spike_times'][unit_row_idx]
    
    # Filter spike times if there are too many to plot clearly (e.g., first 100 seconds)
    st_filtered = st[st < 100] # Show spikes up to 100 seconds
    if len(st_filtered) > 0:
        spike_times_list.append(st_filtered)
        labels.append(f'Unit {unit_id}')
    else:
        print(f"Unit {unit_id} has no spikes before 100s or is empty, skipping.")

if not spike_times_list:
    print("No spike data to plot within the first 100 seconds for selected units.")
    # Create an empty plot or a message
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "No spike data to plot for the selected units/time range.",
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_title('Spike Raster Plot (First 5 Units, up to 100s)')
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
else:
    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times_list, linelengths=0.75, colors='black')
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Time (s)')
    plt.ylabel('Unit ID')
    plt.title('Spike Raster Plot (First 5 Units, up to 100s)')
    plt.xlim(0, 100) # Ensure x-axis limit is set

plt.tight_layout()
plt.savefig('explore/spike_raster.png')
plt.close() # Close the figure

io.close()
print("Plot saved to explore/spike_raster.png")