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


# full_length = 1680
# full_width = 2220


# cells = np.zeros((total_frames, full_length, full_width))
# for t in tqdm(range(total_frames)):
#     image = plt.imread(f"Data/{data}/RawData/{t:0>4}.TIF")
#     smoothed_image = gaussian_filter(image, sigma = 2, mode = 'constant', radius = 2)

#     cells[t] = smoothed_image

full_length = 1680//4
full_width = 2220//4

stitch_length = full_length//3
stitch_width = full_width//4
cut_width = 2220//16*4




cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")[..., :cut_width]

# Normalize boxes
for i in range(3):
    for j in range(4):
        box = cells[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
        box /= box[box>115].mean()
        


# Binning
box_size = 2

n_average = 30
 
length = full_length//box_size
width = cut_width//box_size

bin_threshold = 0.6

classified_pixels = np.zeros((total_frames-2, length, width))
for i in tqdm(range(length)):
    for j in range(width):
        # classified_pixels[:, i,j] = classify_points_binarized(cells, (i+16)*box_size, (j+10)*box_size, box_size, (0,total_frames), threshold)
        # classified_pixels[:, i,j] = classify_points_binarized(cells, i*box_size, j*box_size, box_size, (0,total_frames))
        classified_pixels[:, i,j] = classify_points_normalized(cells, i*box_size, j*box_size, box_size, (0,total_frames))
# np.save(f"Data/{data}/Analysed_data/binarized_method_{bin_threshold}_mix.npy", classified_pixels)
np.save(f"Data/{data}/Analysed_data/normalized_method_{bin_threshold}_mix.npy", classified_pixels)
# plt.imshow(classified_pixels[1], cmap = 'hsv')
# plt.colorbar()