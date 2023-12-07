import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


data = "S-23-11-24"

total_frames = 1300
frame_start = 450
frame_end = 850
cells = np.zeros((total_frames, 1680, 2220))
for i in tqdm(range(total_frames)):
    image = plt.imread(f"Data/{data}/RawData/{i:0>4}.TIF")

    cells[i] = image
np.save("Analysed_data/cells.npy", cells)