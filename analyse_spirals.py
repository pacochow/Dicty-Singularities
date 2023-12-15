import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd


data = "S-23-12-06"
frame_intervals = 15

total_frames = 900
frame_start = 0
frame_end = total_frames
start_time = 3.5

classified_pixels = np.load(f"Data/{data}/Analysed_data/normalized_method_0.6_mix.npy")

smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)
# smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 3)
# pooled = phase_mean_pooling(smoothed)

# winding_numbers = compute_winding_number(smoothed)
# np.save(f"Data/{data}/Analysed_data/winding_numbers.npy", winding_numbers)

positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")
tracking = (positive, negative)

 
filename = f"Data/{data}/Vids/tracking_normalize_trial.mp4" 
animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking)



# filename = f"Data/{data}/Vids/tracking_trial.mp4" 
# create_tracking_animation(pooled, winding_numbers, filename, 40, start_time, frame_intervals, frame_start)
