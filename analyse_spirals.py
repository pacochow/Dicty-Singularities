import numpy as np
from src.animate import *
from src.analysis_utils import *
from src.figures import *
import pandas as pd


data = "Other" 
frame_intervals = 15
frame_start = 0
start_time = 3

classified_pixels = np.load(f"Data/{data}/Analysed_data/normalized_method_0.005_mix.npy")
total_frames = classified_pixels.shape[0]

smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

# winding_numbers = compute_winding_number(smoothed)
# np.save(f"Data/{data}/Analysed_data/winding_numbers.npy", winding_numbers)

# filename = f"Data/{data}/Vids/identifying_normalize_mix.mp4" 
# animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = False)


positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")
tracking = (positive, negative)

filename = f"Data/{data}/Vids/tracking_normalize_mix.mp4" 
animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = False)
 


# filename = f"Data/{data}/Vids/tracking_trial.mp4" 
# create_tracking_animation(pooled, winding_numbers, filename, 40, start_time, frame_intervals, frame_start)


# filename = f"Figs/{data}_phase"
# create_stills(smoothed, filename, [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840], format = (2,8), dims = (30,10), tracking = tracking)
