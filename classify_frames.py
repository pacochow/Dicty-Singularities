import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
from scipy.ndimage import gaussian_filter

data = "S-23-12-06" 
frame_intervals = 15

total_frames = 900
start_time = 3.5


cells = np.zeros((total_frames, 1680, 2220))
for i in tqdm(range(total_frames)):
    image = plt.imread(f"Data/{data}/RawData/{i:0>4}.TIF")
    smoothed_image = gaussian_filter(image, sigma = 2)

    cells[i] = image


# Binning
box_size = 5

n_average = 30
threshold = 10

length = 336
width = 444

bin_threshold = 110

classified_pixels = np.zeros((total_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        # classified_pixels[:, i,j] = classify_points_binarized(cells, (i+16)*box_size, (j+10)*box_size, box_size, (0,total_frames), threshold)
        classified_pixels[:, i,j] = classify_points_binarized(cells, i*box_size, j*box_size, box_size, (0,total_frames), threshold)
# np.save(f"Data/{data}/Analysed_data/floor_method_{n_average}_{threshold}.npy", classified_pixels)
np.save(f"Data/{data}/Analysed_data/binarized_method_{bin_threshold}_{threshold}.npy", classified_pixels)

# plt.imshow(classified_pixels[1], cmap = 'hsv')
# plt.colorbar()