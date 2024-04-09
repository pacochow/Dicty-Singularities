import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
from scipy.ndimage import gaussian_filter
import pandas as pd

data = "F-24-03-14" 
frame_intervals = 4.25


start_time = 3

frame_start = 400
frame_end = 2000

cells = np.zeros((frame_end-frame_start+1, 600, 600))
for i, frame in enumerate(np.arange(frame_start, frame_end+1)):
    image = plt.imread(f"Data/{data}/RawData/{frame:0>4}.tif")
    cells[i] = image


total_frames = cells.shape[0]


# Binning
box_size = 15
stride = 3
length = (cells.shape[1]-box_size)//stride
width = (cells.shape[2]-box_size)//stride
base_threshold = 2.5



# index1 =300
# index2= 450
# time = [0, frame_end-frame_start]

# range1 = (index1, index1+box_size)
# range2 = (index2, index2+box_size)

# raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)

# # Get signal for box averaged over pixels with cells
# signal = raw_signal.mean(axis=1)

# # Smooth signal 
# smoothed_signal = savgol_filter(signal, 25, 2)


# # Find high peaks
# high_peaks = find_peaks(smoothed_signal, distance = 30)[0]

# if len(high_peaks)==0:
#     high_peaks = np.arange(time[0], time[1])

# # Linearly interpolate high peaks to compute floor
# floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])

# # Normalize signal
# diff = floor - smoothed_signal
# diff = savgol_filter(diff, 5, 2)

# # Compute derivatives
# derivatives = [(diff[i+1]-diff[i-1]) for i in range(1, len(diff)-1)]


# # Compute threshold
# diff_peaks = find_peaks(diff, distance = 20)[0]
# if len(diff_peaks)==0:
#     diff_peaks = np.arange(time[0], time[1])
# diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
# threshold = savgol_filter(diff_floor, 100, 2)/2
# threshold = np.maximum(threshold, np.ones(len(threshold))*base_threshold)
# # Normalize diff
# normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]
# plt.figure(figsize = (20,5))
# # plt.plot(signal)
# plt.plot(smoothed_signal)
# plt.plot(floor)


classified_pixels = np.zeros((total_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        classified_pixels[:, i,j] = classify_points_singularity_formation(cells, i*stride, j*stride, box_size, (0,total_frames), base_threshold)

np.save(f"Data/{data}/Analysed_data/normalized_method_{base_threshold}_mix.npy", classified_pixels)

smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

filename = f"Data/{data}/Vids/phase_normalize_mix.mp4" 
animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = False)




# winding_numbers = compute_winding_number_sf(smoothed)
# np.save(f"Data/{data}/Analysed_data/winding_numbers.npy", winding_numbers)

# filename = f"Data/{data}/Vids/identifying_normalize_mix.mp4" 
# animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=winding_numbers, tracking = False)


