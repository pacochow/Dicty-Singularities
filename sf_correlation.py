import numpy as np
import pandas as pd
from src.analysis_utils import *
from src.animate import *
import glob
from tqdm import tqdm
from sklearn.cluster import DBSCAN


data = "F-24-03-13"
frame_intervals = 4.25
frame_start = 400
start_time = 3

files = glob.glob(f"Data/{data}/Analysed_data/normalized*")
classified_pixels = np.load(files[0])
smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

binary = np.logical_and(smoothed<3, smoothed>2.9)
indices = np.where(binary==1)
txy_values = np.stack(indices, axis=-1)
clustering = DBSCAN(eps=3, min_samples = 1).fit(txy_values)


# Get unique labels (including -1 for noise)
unique_labels = np.unique(clustering.labels_)

# # Create a list of colors from the tuple and create a cyclic iterator of colors
# colors = list(plt.cm.tab20.colors)  # Convert tuple to list
# color_cycle = itertools.cycle(colors)

# # Assign a color to each label (cluster) using the cyclic color iterator
# label_color_map = {label: next(color_cycle) for label in unique_labels}
# # Reset color for noise to black, if present in labels
# if -1 in label_color_map:
#     label_color_map[-1] = 'k'

# # Iterate through each time point to plot
# for t in np.arange(0, 1700, 100):
#     # Filter txy_values and labels for the current time
#     current_time_indices = np.where(txy_values[:, 0] == t)
#     current_time_values = txy_values[current_time_indices]
#     current_labels = clustering.labels_[current_time_indices]
    
#     # Plot each cluster with a different color
#     unique_labels = np.unique(current_labels)
#     for label in unique_labels:
#         # Filter points belonging to the current label
#         label_indices = np.where(current_labels == label)
#         points = current_time_values[label_indices]
        
#         color = label_color_map[label]
        
#         # Scatter plot for points
#         plt.scatter(points[:, 1], points[:, 2], s=10, color=color, label=f"Cluster {label}" if label != -1 else "Noise")
    
#     # Avoiding duplicate labels in legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
    
#     plt.title(f"Time: {t}")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()


# Get cluster sizes
cluster_sizes = np.bincount(clustering.labels_)
cluster_sizes_dict = {i: size for i, size in enumerate(cluster_sizes)}
pixel_history = {}

for i, (t, y, x) in enumerate(txy_values):
    if (y, x) not in pixel_history:
        pixel_history[(y, x)] = set()
    pixel_history[(y, x)].add(clustering.labels_[i])


label_distribution = {}

# Iterate over each time point
for t in tqdm(np.arange(binary.shape[0])):
    if len(np.where(txy_values[:, 0]==t)[0])==0:
        label_distribution[t] = 0
        continue
    # Identify which pixels are active at this time point
    active_indices = np.where(txy_values[:, 0] == t)[0]
    active_labels = clustering.labels_[active_indices]


    label_counts = {label: 0 for label in np.unique(active_labels)}

    for label in np.unique(active_labels):
        # For each label, count how many pixels have this label in their history
        for (y, x), history in pixel_history.items():
            if label in history:
                label_counts[label] += 1

    # Store the counts for this time point
    label_distribution[t] = np.max(list(label_counts.values()))/(binary.shape[1]*binary.shape[2])


plt.plot(np.arange(binary.shape[0]), label_distribution.values())
times = [3.5, 4, 4.5, 5, 5.5]
frames = (3600/frame_intervals)*(np.array(times)-(3+frame_intervals*frame_start/3600))
plt.xlabel("Time after starvation (h)")
plt.xticks(frames, times);
plt.ylabel("Proportion of cells correlated")
plt.yscale('log')


# filename = f"Data/{data}/Vids/clustered.mp4" 
# animate_correlation(binary, filename, 40, txy_values, clustering)