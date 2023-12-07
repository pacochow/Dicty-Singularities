import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
import scipy.ndimage as ndimage
from src.animate import *
from scipy.ndimage import gaussian_filter

total_frames = 900
data = "S-23-12-06"
frame_intervals = 15
start_time = 3.5



cells = np.zeros((total_frames, 1680, 2220))
for i in tqdm(range(total_frames)):
    image = plt.imread(f"Data/{data}/RawData/{i:0>4}.TIF")
    
    # Apply gaussian filter
    # image = gaussian_filter(image, sigma=5)
    cells[i] = image

filename = f"Data/{data}/Vids/timelapse.mp4"

create_data_timelapse(cells,filename, 40, start_time, frame_intervals)