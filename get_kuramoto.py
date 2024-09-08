import numpy as np
import pandas as pd
from src.analysis_utils import *
from src.animate import *
import glob
from tqdm import tqdm
import cmath

all_data = {
#     'S-24-04-04': [1877, 4, 1, 3],
#     'S-24-04-05': [2378, 1, 1, 3],
#     'S-24-05-21': [2604, 1, 9, 3],
#     'S-24-05-08-PM': [3353, 5, 6, 3],
#     'S-24-04-02': [3574, 6, 3, 3],
#     'S-24-05-10-PM': [3721, 2, 6, 3],
#     'S-24-05-09-PM': [3815, 10, 5, 3],
#     'S-24-01-12-PM': [4091, 7, 4, 3],
#     'S-24-04-12': [4183, 4, 2, 3],
#     'S-24-04-03': [4460, 7, 3, 3],
#     'S-24-02-07-AM': [4493, 7, 3, 3],
#     'S-24-02-02': [4787, 12, 3, 3],
#     'S-24-01-24': [5047, 4, 4, 3],
#     'S-24-01-23': [5080, 7, 4, 3],
#     'S-24-01-30-PM': [5381, 14, 3, 3+1/6],
#     'S-24-01-17-PM': [5388, 4, 5, 3],
#     'S-24-02-06-PM': [5394, 15, 0, 3],
#     'S-24-05-24-AM': [5446, 12, 1, 3],
#     'S-24-02-07-PM': [5491, 12, 0, 3],
    'S-24-01-25': [5667, 8, 2, 3],
#     'S-24-04-23-PM': [5716, 17, 1, 3],
#     'S-24-01-30-AM': [5772, 6, 0, 3],
#     'S-24-04-25-PM': [5810, 15, 0, 3],
#     'S-24-04-25-AM': [5823, 4, 0, 3],
#     'S-24-02-06-AM': [5970, 8, 3, 3+1/3],
#     'S-24-02-01': [5994, 9, 5, 3],
#     'S-24-01-11': [6031, 13, 2, 3],
#     'S-24-05-23-FML': [6105, 20, 1, 3],
#     'S-24-05-23-AM': [6111, 13, 3, 3],
#     'S-24-01-12-AM': [6216, 13, 1, 3],
#     'S-24-01-16': [6302, 13, 0, 3],
#     'S-24-01-17-AM': [6358, 8, 1, 3],
#     'S-24-05-09-AM': [6372, 14, 7, 3],
#     'S-24-01-31': [6467, 13, 2, 3],
#     'S-24-01-10': [6586, 11, 0, 3],
#     'S-24-04-23-AM': [6662, 14, 1, 3],
#     'S-24-04-24-AM': [6902, 24, 0, 3],
#     'S-24-05-07': [7069, 20, 0, 2+5/6],
#     'S-24-05-23-PM': [7321, 14, 2, 3],
#     'S-24-01-26': [7376, 13, 1, 3],
#     'S-24-05-10-AM': [7713, 23, 1, 3],
#     'S-24-05-22-PM': [8401, 20, 1, 3],
#     'S-24-05-08-AM': [8648, 22, 0, 2+5/6],
#     'S-24-04-11': [8704, 16, 1, 3],
#     'S-24-05-22-AM': [9221, 26, 1, 3],
    # "C-24-06-01": [7781, 21, 0, 2.5, 0.25],
    # "C-24-06-02": [7092, 29, 0, 2.5, 0.25],
    # "C-24-06-04": [7673, 35, 0, 2.5, 0.25], 
    # "C-24-06-05": [8846, 22, 0, 2, 0.25]
    # "C-24-05-24": [6644, 8, 0, 2+23/60, 0.5],
    # "C-24-05-29": [5391, 8, 0, 2, 0.5],
    # "C-24-05-30-0.5": [6601, 14, 0, 2+1/12, 0.5], 
    # "C-24-05-31-0.5": [6875, 11, 2, 2.5, 0.5],
    # "C-24-05-22": [7269, 5, 1, 2.5, 1],
    # "C-24-05-28": [6745, 11, 0, 2, 1],
    # "C-24-05-30-1": [7038, 10, 0, 2, 1],
    # "C-24-05-31-1": [6690, 13, 0, 2, 1]
    }

def complex_exp(x):
    return cmath.exp(x*1j)

frame_intervals = 15
for data in all_data:
# data = "S-24-05-22-AM"    

    files = glob.glob(f"Data/{data}/Analysed_data/normalized*")
    classified_pixels = np.load(files[0])
    smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)
    smoothed = smoothed[:, 20:smoothed.shape[1]-20, 20:smoothed.shape[2]-20]



    # Vectorize the function
    vectorized_exp = np.vectorize(complex_exp)

    local_box = 9
    # mask = np.logical_or(smoothed > np.pi/6, smoothed<-np.pi/6)

    kuramoto = np.zeros((smoothed.shape[0], smoothed.shape[1] - local_box, smoothed.shape[2] - local_box))
    for y in tqdm(range(smoothed.shape[1] - local_box)):
        for x in range(smoothed.shape[2] - local_box):
            for t in range(smoothed.shape[0]):
                # if mask[t, y, x]:
                exp_smoothed = vectorized_exp(smoothed[t, y:y+local_box, x:x+local_box])
                kuramoto[t, y, x] = abs(np.nanmean(exp_smoothed))
                # else:
                #     kuramoto[t, y, x] = np.nan

    mean_kuramoto = np.nanmean(kuramoto, axis=(1, 2))

    plt.plot(mean_kuramoto[:600])
    plt.ylabel("Kuramoto parameter")
    times = [3, 3.5, 4, 4.5, 5, 5.5]
    frames = (3600/frame_intervals)*(np.array(times)-3)
    plt.xlabel("Time after starvation (h)")
    plt.xticks(frames, times);
    plt.show()
    np.save(f"Data/{data}/Analysed_data/kuramoto.npy", mean_kuramoto)