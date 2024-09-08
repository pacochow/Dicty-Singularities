import numpy as np
import pandas as pd
from src.analysis_utils import *
from src.animate import *
import glob
from tqdm import tqdm

data = "S-24-01-25"

files = glob.glob(f"Data/{data}/Analysed_data/normalized*")
classified_pixels = np.load(files[0])
smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

height = smoothed.shape[1]
width = smoothed.shape[2]
smoothed = smoothed[:, 20:height-20, 20:width-20]
height = smoothed.shape[1]
width = smoothed.shape[2]
left=np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, 1:height-1, :width-2]))
right=np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, 1:height-1, 2:width]))
top=np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, :height-2, 1:width-1]))
bottom=np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, 2:height, 1:width-1]))
top_left = np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, :height-2, :width-2]))
top_right = np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, :height-2, 2:width]))
bottom_left = np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, 2:height, :width-2]))
bottom_right = np.abs(phase_difference(smoothed[:, 1:height-1, 1:width-1], smoothed[:, 2:height, 2:width]))
sync = np.array([left, right, top, bottom, top_left, top_right, bottom_left, bottom_right]).mean(axis=0)
sync_mean = sync.mean(axis = (1,2))
sync_mean_smoothed = savgol_filter(sync_mean, 80, 2)
desynchronised_frame = sync_mean_smoothed[:400].argmax()
plt.plot(sync_mean[:600])
plt.show()





def phase_difference(phase1, phase2):
    mask = np.logical_or(np.isnan(phase1), np.isnan(phase2))
    phase1 = np.where(mask, 0, phase1)
    phase2 = np.where(mask, 0, phase2)
    diff = phase2 - phase1
    return np.where(mask, np.nan, np.arctan2(np.sin(diff), np.cos(diff)))

height = smoothed.shape[1]
width = smoothed.shape[2]
smoothed = smoothed[:, 20:height-20, 20:width-20]
height = smoothed.shape[1]
width = smoothed.shape[2]

# Boolean indexing to select only elements where smoothed > 0
smoothed_positive = np.where(smoothed > 0, smoothed, np.nan)

left = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, 1:height-1, :width-2]))
right = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, 1:height-1, 2:width]))
top = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, :height-2, 1:width-1]))
bottom = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, 2:height, 1:width-1]))
top_left = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, :height-2, :width-2]))
top_right = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, :height-2, 2:width]))
bottom_left = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, 2:height, :width-2]))
bottom_right = np.abs(phase_difference(smoothed_positive[:, 1:height-1, 1:width-1], smoothed_positive[:, 2:height, 2:width]))

sync = np.array([left, right, top, bottom, top_left, top_right, bottom_left, bottom_right]).mean(axis=0)
sync_mean = np.nanmean(sync, axis=(1, 2))
plt.plot(sync_mean[:600])
plt.show()