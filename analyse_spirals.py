import numpy as np
from src.animate import *
from src.analysis_utils import *
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


data = "S-23-12-06"
frame_intervals = 15

total_frames = 900
frame_start = 0
frame_end = total_frames
start_time = 3.5

classified_pixels = np.load(f"Data/{data}/Analysed_data/binarized_method_110_10.npy")

smoothed = phase_gaussian_smoothing(classified_pixels)
pooled = phase_mean_pooling(smoothed)

winding_numbers = compute_winding_number(pooled)
filename = f"Data/{data}/Vids/tracking_trial.mp4" 
animate_processed_data(pooled, filename, 40, start_time, frame_intervals, frame_start, tracking=winding_numbers)



# filename = f"Data/{data}/Vids/tracking_trial.mp4" 
# create_tracking_animation(pooled, winding_numbers, filename, 40, start_time, frame_intervals, frame_start)
