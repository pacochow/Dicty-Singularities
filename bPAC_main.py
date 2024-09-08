import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
import pandas as pd
import trackpy as tp

data = "A-24-05-17" 
frame_intervals = 20
start_time = 3
frame_start = 0
# activation_time = 5
# activation_frame = int(frame_start+(activation_time-start_time)*3600/frame_intervals)
activation_frame = 329
pre_activation_frame = 0
post_activation_frame = 36

full_length = 2220//4
full_width = 1680//4

stitch_length = full_length//4
stitch_width = full_width//3



cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
cells/=cells[:500].mean(axis=0)


cells_pre=cells[:activation_frame-pre_activation_frame]
pre_frames = cells_pre.shape[0]

# Binning
box_size = 5
length = (full_length-box_size)//2
width = (full_width-box_size)//2
base_threshold = 0.01

# Normalize boxes
for i in range(4):
    for j in range(3):
        box = cells_pre[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box_masked = np.where(box<1.2, box, np.nan)
        box_masked = np.where(box_masked>0.9, box_masked, np.nan)
        box_mean_masked = np.nanmean(box_masked, axis = (1,2))[:, np.newaxis, np.newaxis]
        box /= box_mean_masked


# Smooth cell signal
for i in tqdm(range(cells_pre.shape[1])):
    for j in range(cells.shape[2]):
        cells_pre[:, i, j] = savgol_filter(cells_pre[:, i, j], 10, 2)


classified_pixels_early = np.zeros((pre_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):

        classified_pixels_early[:, i,j] = classify_points_time_normalized(cells_pre, i*2, j*2, box_size, (0,pre_frames), base_threshold)




cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
cells/=cells[:500].mean(axis=0)
cells_post = cells[activation_frame+post_activation_frame:]
post_frames = cells_post.shape[0]

# Normalize boxes
for i in range(4):
    for j in range(3):
        box = cells[activation_frame+post_activation_frame:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box /= box[:500].mean()



# Smooth cell signal
for i in tqdm(range(cells.shape[1])):
    for j in range(cells.shape[2]):
        cells_post[:, i, j] = savgol_filter(cells_post[:, i, j], 10, 2)


classified_pixels_late = np.zeros((post_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        classified_pixels_late[:, i,j] = classify_points_time_normalized(cells_post, i*2, j*2, box_size, (0,post_frames), base_threshold)

activate = np.ones((pre_activation_frame+post_activation_frame, length, width))*np.pi
classified_pixels = np.concatenate((classified_pixels_early, activate, classified_pixels_late), axis = 0)
np.save(f"Data/{data}/Analysed_data/normalized_method_{base_threshold}_mix.npy", classified_pixels)


# Identify singularities
smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

winding_numbers = compute_winding_number(smoothed)
np.save(f"Data/{data}/Analysed_data/winding_numbers.npy", winding_numbers)

# Track all singularities
positives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")>0.9
negatives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")<-0.9
length = positives.shape[1]
width = positives.shape[2]

# Search for 7 pixel wide particles
f_p = tp.batch(positives, 7, processes = 1)

# Link if particle if next time frame is wthin 10 particles away and allow particle to be remembered for 15 frames if disappeared
t_p = tp.link(f_p, 10, memory = 15)

# Filter particle trajectories that are only present for 5 frames
t1_p = tp.filter_stubs(t_p, 5)

f_n = tp.batch(negatives, 7, processes = 1)
t_n = tp.link(f_n, 10, memory = 15)
t1_n = tp.filter_stubs(t_n, 5)

# Drop columns
t1_p = t1_p.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis = 1)
t1_n = t1_n.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis = 1)

tracking = (t1_p, t1_n)


filename = f"Data/{data}/Vids/tracking_normalize_mix_unfiltered.mp4" 
animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = True)

# # Filter singularities
# positive, negative = filter_singularities(t1_p, t1_n, [10, 5, 10, 25], [40, 40, 40, 40], 400, 400, 600)

# # Save singularities
# positive.to_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
# negative.to_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")

# positive['spin'] = '+'
# negative['spin'] = '-'
# negative['particle'] += positive['particle'].max()

# singularities = pd.concat([positive, negative], ignore_index=True)
# singularities.to_pickle(f"Data/{data}/Analysed_data/singularities.pkl")


# tracking = (positive, negative)
# filename = f"Data/{data}/Vids/tracking_normalize_mix.mp4" 
# animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = False)
 