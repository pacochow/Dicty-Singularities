import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
from src.figures import *
from scipy.ndimage import gaussian_filter

data = "S-24-02-01" 
frame_intervals = 15


start_time = 3

full_length = 2220//4
full_width = 1680//4


stitch_length = full_length//4
stitch_width = full_width//3




cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
cells/=cells[:500].mean(axis=0)
total_frames = cells.shape[0]


# Binning
box_size = 5
length = (full_length-box_size)//2
width = (full_width-box_size)//2
base_threshold = 0.01

# Normalize boxes
for i in range(4):
    for j in range(3):
        box = cells[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box_masked = np.where(box<1.2, box, np.nan)
        box_masked = np.where(box_masked>0.9, box_masked, np.nan)
        box_mean_masked = np.nanmean(box_masked, axis = (1,2))[:, np.newaxis, np.newaxis]
        box /= box_mean_masked


# Smooth cell signal
for i in tqdm(range(cells.shape[1])):
    for j in range(cells.shape[2]):
        cells[:, i, j] = savgol_filter(cells[:, i, j], 10, 2)


classified_pixels_early = np.zeros((total_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        # classified_pixels[:, i,j] = classify_points_binarized(cells, i*box_size, j*box_size, box_size, (0,total_frames))
        classified_pixels_early[:, i,j] = classify_points_time_normalized(cells, i*2, j*2, box_size, (0,total_frames), base_threshold)




cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
cells/=cells[:500].mean(axis=0)



# Normalize boxes
for i in range(4):
    for j in range(3):
        box = cells[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box /= box[:500].mean()


# Smooth cell signal
for i in tqdm(range(cells.shape[1])):
    for j in range(cells.shape[2]):
        cells[:, i, j] = savgol_filter(cells[:, i, j], 10, 2)


classified_pixels_late = np.zeros((total_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        # classified_pixels[:, i,j] = classify_points_binarized(cells, i*box_size, j*box_size, box_size, (0,total_frames))
        classified_pixels_late[:, i,j] = classify_points_time_normalized(cells, i*2, j*2, box_size, (0,total_frames), base_threshold)
# np.save(f"Data/{data}/Analysed_data/binarized_method_{bin_threshold}_mix.npy", classified_pixels)
    
transition = 150
classified_pixels = np.concatenate((classified_pixels_early[:transition], classified_pixels_late[transition:]), axis = 0)
np.save(f"Data/{data}/Analysed_data/normalized_method_{base_threshold}_mix.npy", classified_pixels)

