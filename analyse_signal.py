import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import math
from scipy.ndimage import uniform_filter
import seaborn as sns


data = "Other"

total_frames = 900
full_length = 2220//4
full_width = 1680//4


stitch_length = full_length//4
stitch_width = full_width//3


cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
cells/=cells[:500].mean(axis=0)

# Normalize boxes
for i in range(4):
    for j in range(3):
        box = cells[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box_masked = np.where(box<1.2, box, np.nan)
        box_masked = np.where(box_masked>0.9, box_masked, np.nan)
        box_mean_masked = np.nanmean(box_masked, axis = (1,2))[:, np.newaxis, np.newaxis]
        box /= box_mean_masked


# # Normalize boxes
# for i in range(4):
#     for j in range(3):
#         box1 = cells[:150, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
#         line = cells[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width].mean(axis = (1,2))
#         smoothed_line = savgol_filter(line, 80,2)

#         change = np.argmin(np.abs(smoothed_line-box1.mean()))
#         change2 = np.abs(line-box1.mean())
#         sorted_indices = [index for index, value in sorted(enumerate(change2), key=lambda pair: pair[1])]
#         change3 = next((index for index in sorted_indices if abs(index-change) <= 5), None)
#         print(line[change3], box1.mean())
#         print(change3, change)
#         # box_masked = np.where(box<1.2, box, np.nan)
#         # box_mean_masked = np.nanmean(box_masked, axis = (1,2))[:, np.newaxis, np.newaxis]
#         box2 = cells[:change3+1, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]

#         box2 /= line[:change3+1][:, np.newaxis, np.newaxis]
        
#         box3 = cells[change3+1:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
#         box3 /= box1.mean()
        

        

# Smooth cell signal
for i in tqdm(range(cells.shape[1])):
    for j in range(cells.shape[2]):
        cells[:, i, j] = savgol_filter(cells[:, i, j], 10, 2)



# Binning
index1 = 100
index2 = 59
box_size = 5
range1 = (index1, index1+box_size)
range2 = (index2, index2+box_size)
time = (0, total_frames)
threshold = 3
base_threshold = 0.005
raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)


# Get signal for box averaged over pixels with cells
signal = raw_signal.mean(axis=1)

# Smooth signal 
smoothed_signal = savgol_filter(signal, 10, 2)


# Find high peaks
high_peaks = find_peaks(smoothed_signal, distance = 20)[0]

if len(high_peaks)==0:
    high_peaks = np.arange(time[0], time[1])

# Linearly interpolate high peaks to compute floor
floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])

# Normalize signal
diff = floor - smoothed_signal
diff = savgol_filter(diff, 5, 2)

# Compute derivatives
derivatives = [(diff[i+1]-diff[i-1]) for i in range(1, len(diff)-1)]


# Compute threshold
diff_peaks = find_peaks(diff, distance = 20)[0]
if len(diff_peaks)==0:
    diff_peaks = np.arange(time[0], time[1])
diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
threshold = savgol_filter(diff_floor, 50, 2)/2
threshold = np.maximum(threshold, np.ones(len(threshold))*base_threshold)
# Normalize diff
normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]


# Compute phase angles
angles = np.array([(math.atan2(normalized_signal[i],derivatives[i])+np.pi/2)%(2*np.pi) for i in range(len(smoothed_signal)-2)])

fig = plt.figure(figsize=(20, 7))

# Creating a grid for subplots
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[:, 1])
ax1.scatter(derivatives, normalized_signal, s=2)
ax1.axvline(0, c='black')
ax1.axhline(0, c='black')
ax1.set_xlabel("$F'$", size = 15)
ax1.set_ylabel("$F-F_{threshold}$", size = 15)
ax1.set_aspect('equal', 'box')

ax2 = fig.add_subplot(gs[0, 0])
ax2.plot(smoothed_signal, label='$F$')
ax2.plot(floor, label='$F_{floor}$')
ax2.set_xticklabels([])
ax2.legend()

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(diff, label='$F-F_{floor}$')
ax3.plot(threshold, label='$F_{threshold}$')
ax3.set_xlabel("Frame", size = 15)
ax3.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize = (10,4))
plt.plot(diff, linewidth = 2, c='black')
times = [3, 3.5, 4, 4.5, 5, 5.5, 6]
frames = (3600/15)*(np.array(times)-3)
plt.xticks(frames, times)
plt.xlabel("Time after starvation (h)", fontsize = 12);
plt.ylabel("F", fontsize = 12)
sns.despine()
plt.show()

plt.figure(figsize = (5,5))
plt.scatter(derivatives, normalized_signal, s=2)
plt.axvline(0, c='black')
plt.axhline(0, c='black')
plt.xticks([-0.03, 0.03])
plt.yticks([-0.03, 0.03])
plt.xlabel("F'")
plt.ylabel("F")

filter = np.concatenate([np.ones(5)*(-1),np.ones(9)*1, np.ones(5)*(-1)])
convolve = np.convolve(filter, diff, 'same')

filter2 = np.concatenate([np.ones(10)*(-1),np.ones(8)*(1)])
convolve2 = np.convolve(filter2, derivatives, 'same')

# plt.figure(figsize = (20,5))
# plt.plot(diff)
# plt.plot(derivatives)
# plt.plot(convolve, c='black')
# plt.axhline(0.05)


# visualize_pixel_evolution(cells, range1, range2, time)



