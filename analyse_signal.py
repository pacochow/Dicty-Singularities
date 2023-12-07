import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import math


data = "S-23-12-06"

total_frames = 900
cells = np.zeros((total_frames, 1680, 2220))
for i in tqdm(range(total_frames)):
    image = plt.imread(f"Data/{data}/RawData/{i:0>4}.TIF")
    smoothed_image = gaussian_filter(image, sigma = 2, mode = 'constant', radius = 2)

    cells[i] = smoothed_image

# Binning
index1 = 300
index2 = 1200
box_size = 5
range1 = (index1, index1+box_size)
range2 = (index2, index2+box_size)
time = (0, total_frames)
threshold = 10


signal = np.sort(cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2), axis = 1)[:, ::-1][:, :30].mean(axis = 1)
smoothed_signal = savgol_filter(signal, 13, 2)
derivatives = [(smoothed_signal[i+1]-smoothed_signal[i-1]) for i in range(1, len(smoothed_signal)-1)]

midline = savgol_filter(signal, 80, 2)
normalized_signal = (smoothed_signal - midline)[1:len(smoothed_signal)-1]
angles = [math.atan2(derivatives[i], normalized_signal[i]) for i in range(len(smoothed_signal)-2)]

plt.figure(figsize = (20,5))
plt.plot(signal)
plt.show()

smoothed_signal = savgol_filter(signal, 5, 2)

# Find high peaks
high_peaks = find_peaks(smoothed_signal, distance = 1)[0]

raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)

# Compute which pixels have cells
binarized = np.apply_along_axis(cell_or_not, 1, raw_signal)

# Set mask where there are no cells and compute mean 
masked = np.where(binarized == 1, raw_signal, np.nan)
signal = np.nanmean(masked, axis = 1)
signal = np.nan_to_num(signal)

smoothed_signal = savgol_filter(signal, 5, 2)
derivatives = [(smoothed_signal[i+1]-smoothed_signal[i-1]) for i in range(1, len(smoothed_signal)-1)]

# Find high peaks
high_peaks = find_peaks(smoothed_signal, distance = 1, height = 1)[0]

if len(high_peaks)==0:
    high_peaks = np.arange(time[0], time[1])

# Linearly interpolate high peaks to compute floor
floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])
diff = floor - smoothed_signal
diff = savgol_filter(diff, 5, 2)
normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]

angles = np.array([(math.atan2(derivatives[i], normalized_signal[i]+np.pi)%(2*np.pi)) for i in range(len(smoothed_signal)-2)])
angles[signal[1:len(smoothed_signal)-1] == 0] = np.nan


# visualize_pixel_evolution(cells, range1, range2, time)


# plt.figure(figsize = (20,5))

# plt.scatter(low_peaks, normalized_signal[low_peaks], s = 20, c = 'r')
# plt.scatter(high_peaks, normalized_signal[high_peaks], s = 20, c='y')
# plt.scatter(increasing_indices, normalized_signal[increasing_indices], s = 10, c = 'g')
# plt.scatter(decreasing_indices, normalized_signal[decreasing_indices], s = 10, c = 'black')
# plt.scatter(low_peaks, normalized_signal[low_peaks], s = 20, c = 'r')
# plt.scatter(high_peaks, normalized_signal[high_peaks], s = 20, c='y')
# plt.plot(normalized_signal, alpha = 0.5)



