import numpy as np
from src.animate import *
from src.analysis_utils import *
from src.figures import *
import pandas as pd
import glob

data = "B-24-08-28"
frame_intervals = 20
frame_start = 0
start_time = 3

files = glob.glob(f"Data/{data}/Analysed_data/normalized*")
classified_pixels = np.load(files[0])

# raw = np.load(f"Data/{data}/Analysed_data/pooled_cells.npy")
# raw/=raw[:500].mean(axis=0)
# full_length = 2220//4
# full_width = 1680//4

# stitch_length = full_length//4
# stitch_width = full_width//3

# # Normalize boxes
# for i in range(4):
#     for j in range(3):
#         box = raw[:, i*stitch_length:(i+1)*stitch_length, j*stitch_width:(j+1)*stitch_width]
#         box_masked = np.where(box<1.2, box, np.nan)
#         box_masked = np.where(box_masked>0.9, box_masked, np.nan)
#         box_mean_masked = np.nanmean(box_masked, axis = (1,2))[:, np.newaxis, np.newaxis]
#         box /= box_mean_masked


# # Smooth cell signal
# for i in tqdm(range(raw.shape[1])):
#     for j in range(raw.shape[2]):
#         raw[:, i, j] = savgol_filter(raw[:, i, j], 10, 2)



# total_frames = min(classified_pixels.shape[0], raw.shape[0])

smoothed = phase_gaussian_smoothing(classified_pixels, sigma = 1.5)

# winding_numbers = compute_winding_number(smoothed)
# np.save(f"Data/{data}/Analysed_data/winding_numbers.npy", winding_numbers)

# filename = f"Data/{data}/Vids/identifying_normalize_mix.mp4" 
# animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = False)

singularities = pd.read_pickle(f"Data/{data}/Analysed_data/singularities.pkl")
positive = singularities[singularities['spin']=='+']
negative = singularities[singularities['spin']=='-']
tracking = (positive, negative)

filename = f"Data/{data}/Vids/tracking_normalize_mix.mp4" 
animate_processed_data(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = False)
 
# filename = f"Data/{data}/Vids/vid_fig.mp4" 
# animate_processed_data_fig(smoothed, filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = False)

# filename = f"Data/{data}/Vids/vid_with_raw_fig.mp4" 
# animate_processed_data_with_raw_fig(raw[2:total_frames+2], smoothed[:total_frames], filename, 40, start_time, frame_intervals, frame_start, identifying=False, tracking = tracking, coordinates = False)




# filename = f"Data/{data}/Vids/tracking_trial.mp4" 
# create_tracking_animation(pooled, winding_numbers, filename, 40, start_time, frame_intervals, frame_start)


# filename = f"Figs/{data}_phase"
# create_stills(smoothed, filename, [0, 120, 240, 360, 480, 600, 720], format = (1,7), dims = (30,10), tracking = tracking)

### FIG 2A
# raw = plt.imread(f"Figs/520.tif")
# plt.imshow(raw[64:-120, 80:-120], cmap = 'gray', vmin = 90, vmax = 180)
# plt.axis('off')
# plt.axhline(1970, c='white', xmin=0.8, xmax = 0.908)
# plt.savefig(f"Figs/{data}_raw_520.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# frame = 520
# plt.imshow(smoothed[520, 8:-15, 10:-15], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axhline(243, c='white', xmin = 0.80, xmax=0.911)
# plt.axis('off')
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi=300)

### FIG 2C
# frame = 120
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_n[particle], marker = 'o')
# plt.axhline(265, c='white', xmin = 0.80, xmax=0.911)
# # plt.xlim([10, 192])
# # plt.ylim([260, 8])
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)

# # CLOSEUP
# plt.imshow(smoothed[510, 100:130, 70:100], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# plt.savefig(f"Figs/{data}_510_closeup.png", bbox_inches = 'tight', pad_inches = 0, dpi=300)


# PAIRWISE ANNIHILATION CLOSEUP
# frame = 432
# plt.imshow(smoothed[frame], cmap = 'twilight')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 300, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 300, color=colormap_n[particle], marker = 'o')
# plt.xlim([105,140])
# plt.ylim([95, 130])
# plt.axis('off')
# plt.savefig(f"Figs/{data}_pair_1_{frame}.png", bbox_inches = 'tight', pad_inches = 0)



# frame = 476
# plt.imshow(smoothed[frame], cmap = 'twilight')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 300, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 300, color=colormap_n[particle], marker = 'o')
# plt.xlim([90,130])
# plt.ylim([165, 205])
# plt.axis('off')
# plt.savefig(f"Figs/{data}_pair_2_{frame}.png", bbox_inches = 'tight', pad_inches = 0)


# FIG 3A
# frame = 227
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_n[particle], marker = 'o')
# plt.ylim([265, 5])
# plt.xlim([10, 190])
# plt.axhline(250, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)

# FIG 3C 3h B-24-04-12
# frame = 0
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_n[particle], marker = 'o')
# plt.ylim([270, 10])
# plt.xlim([10, 190])
# plt.ylim([250, 10])
# plt.xlim([12, 190])
# plt.axhline(240, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# FIG 3C 3.5h B-24-05-16
# frame = 480
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_n[particle], marker = 'o')
# plt.ylim([260, 10])
# plt.xlim([10, 200])
# plt.ylim([250, 10])
# plt.xlim([12, 190])
# plt.axhline(240, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)

# FIG 3C 4h B-24-04-11
# frame = 0
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_n[particle], marker = 'o')
# plt.ylim([250, 10])
# plt.xlim([12, 200])
# plt.ylim([250, 10])
# plt.xlim([12, 190])
# plt.axhline(240, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# FIG 3C 5h B-24-04-18-AM
# frame = 590
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 80, color=colormap_n[particle], marker = 'o')
# plt.ylim([260, 10])
# plt.xlim([10, 200])
# plt.ylim([250, 10])
# plt.xlim([12, 190])
# plt.axhline(240, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# FIG 3E A-24-05-17
# frame = 315
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 50, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 50, color=colormap_n[particle], marker = 'o')
# plt.ylim([255, 10])
# plt.xlim([15, 193])
# plt.axhline(245, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)
# plt.show()

# FIG 3E A-24-05-02
# frame = 400
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_n[particle], marker = 'o')
# plt.ylim([260, 10])
# plt.xlim([5, 180])
# plt.axhline(250, c='white', xmin = 0.80, xmax=0.911)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# FIG 3E A-24-05-16
# frame = 387
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_n[particle], marker = 'o')
# plt.ylim([260, 10])
# plt.xlim([5, 200])
# plt.axhline(245, c='white', xmin = 0.80, xmax=0.911)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)



# FIG 4C C-24-06-01
# frame = 600
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_n[particle], marker = 'o')
# plt.ylim([245, 8])
# plt.xlim([10, 195])
# plt.axhline(235, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)

# FIG 4C C-24-05-30-0.5
# frame = 0
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_n[particle], marker = 'o')
# plt.ylim([245, 8])
# plt.xlim([10, 195])
# plt.axhline(235, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)


# FIG 4C C-24-05-28
# frame = 600
# plt.imshow(smoothed[frame], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
# plt.axis('off')
# colormap_p, _ = create_tracking_colormap(positive)
# colormap_n, _ = create_tracking_colormap(negative)

# for particle in positive[positive['frame'] == frame]['particle']:
#     particle_data = positive[(positive['frame'] == frame) & (positive['particle'] == particle)]
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_p[particle], marker = 's')
# for particle in negative[negative['frame'] == frame]['particle']:
#     particle_data = negative[(negative['frame'] == frame) & (negative['particle'] == particle)]  
#     plt.scatter(particle_data['x'], particle_data['y'], s = 60, color=colormap_n[particle], marker = 'o')
# plt.ylim([245, 8])
# plt.xlim([10, 195])
# plt.axhline(235, c='white', xmin = 0.80, xmax=0.911, linewidth = 3)
# plt.savefig(f"Figs/{data}_{frame}.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)
