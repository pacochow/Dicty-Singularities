import numpy as np
import pandas as pd
from src.analysis_utils import *
from src.animate import *
import glob
from tqdm import tqdm


data = "S-24-01-25"
frame_intervals = 15
start_time = 3
positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")


files = glob.glob(f"Data/{data}/Analysed_data/normalized*")
classified_pixels = np.load(files[0])
smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

periods = np.full(smoothed.shape, np.nan)
n_waves = np.full(smoothed.shape, np.nan)

for y in tqdm(range(smoothed.shape[1])):
    for x in range(smoothed.shape[2]):
        signal = smoothed[:, y, x]

        # For peaking neighbour, find time to next peak
        peaks = find_peaks(signal, height = 2, distance = 10)[0]

        # If no peaks are above 2
        if len(peaks) == 0:
                continue
        else:
                for t in range(smoothed.shape[0]):
                        diff = peaks - t

                        # Find closest peak
                        index_min = np.abs(diff).argmin()

                        # Count number of waves experienced at that point
                        if t < peaks[index_min]:  
                                n_waves[t, y, x] = index_min
                        else:
                                n_waves[t, y, x] = index_min + 1

                        # If closest peak is first peak
                        if index_min == 0 and t < peaks[index_min]:
                                continue

                        # If closest peak is at last peak
                        if index_min==len(diff)-1 and t >= peaks[index_min]:
                                break
 
                        # Compute time between closest peak and peak after that
                        if t < peaks[index_min]:
                                frames_to_next_peak = peaks[index_min]-peaks[index_min-1]
                        else:   
                                frames_to_next_peak = peaks[index_min+1]-peaks[index_min]
                        
                        per = frames_to_next_peak*15/60 # periodicity in minutes
                        periods[t, y, x] = per

# normalize = np.nanquantile(periods, 0.9, axis=(1,2))
# normalize = np.expand_dims(normalize, axis = (1,2))
# normalized_periods = periods/normalize

# filename = f"Data/{data}/Vids/period_analysis.mp4" 
# animate_periods(periods, filename , 40, start_time, frame_intervals, tracking=[positive, negative])

mean = n_waves.mean(axis = (1,2)).reshape(-1, 1, 1)
n_waves=n_waves/mean

filename = f"Data/{data}/Vids/waves_experienced.mp4" 
animate_periods(n_waves, filename , 40, start_time, frame_intervals, tracking=[positive, negative])

 