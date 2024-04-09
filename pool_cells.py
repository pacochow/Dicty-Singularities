import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.figures import *


data = "S-24-04-03 "

total_frames = 900
frame_start = 0
cells = np.zeros((total_frames, 2220, 1680))
for i in tqdm(range(total_frames)):
    image = plt.imread(f"Data/{data}/RawData/{i:0>4}.tif")
    cells[i] = image

_, M, N = cells.shape
K = 4
L = 4

MK = M // K
NL = N // L
pooled = cells[:, :MK*K, :NL*L].reshape(total_frames, MK, K, NL, L).mean(axis=(2, 4))
np.save(f"Data/{data}/Analysed_data/pooled_cells.npy", pooled)




# filename = f"Data/{data}/Vids/raw_data_normalized_smoothed.mp4"
# create_data_timelapse(cells, filename, 40, start_time, frame_intervals)

# cells = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
# cells/=cells[:500].mean(axis=0)
# filename = f"Figs/{data}_raw_data"
# create_stills(cells, filename, [0, 120, 240, 360, 480, 600, 720, 840], format = (1,8), dims = (15,7))


