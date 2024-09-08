
import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd
from scipy.signal import savgol_filter
import glob
from tqdm import tqdm


datasets = {
    "C-24-06-01": [0.25, 2.5],
    "C-24-06-02": [0.25, 2.5],
    "C-24-06-04": [0.25, 2.5],
    "C-24-06-05": [0.25, 2],
    "C-24-05-24": [0.5, 2+23/60],
    "C-24-05-29": [0.5, 2],
    "C-24-05-30-0.5": [0.5, 2+1/12],
    "C-24-05-31-0.5": [0.5, 2.5],
    # "C-24-05-22": [1, 2.5],
    "C-24-05-28": [1, 2],
    "C-24-05-30-1": [1, 2],
    "C-24-05-31-1": [1, 2]
    }



frame_intervals = 15

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
caffeine_color = {
    0.25: 'tab:red',
    0.5: 'tab:green',
    1: 'tab:blue'
    }

for i, data in enumerate(datasets):
    
    singularities = pd.read_pickle(f"Data/{data}/Analysed_data/singularities.pkl")
    positive = singularities[singularities['spin']=='+']
    negative = singularities[singularities['spin']=='-']
    start_time = datasets[data][1]

    s = get_n_singularities(data).astype(float)

    mean_s = smooth_singularity_time_series(s[:600], since_birth=False)

    # color = colors[i % len(colors)]
    color = caffeine_color[datasets[data][0]]
    times = np.arange(start_time, start_time+(singularities['frame'].max()-10)/(3600/frame_intervals), 1/(3600/frame_intervals))
    plt.scatter(times, s[:len(times)], s=1, alpha=0.5, color = color)
    plt.plot(times[:len(mean_s[:len(times)-40])], mean_s[:len(times)-40], alpha = 0.8, linewidth = 4, color = color, label = f"{datasets[data][0]}")
# plt.axhline(14, linestyle = '--', color = 'black', alpha = 0.5, label = '0')
plt.xlabel("Time after starvation (h)", fontsize = 12) 
plt.ylabel("Number of singularities", fontsize = 12)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()



all_control_data = {
    'S-24-04-04': [1877, 4, 1, 3],
    'S-24-04-05': [2378, 1, 1, 3],
    'S-24-05-21': [2604, 1, 9, 3],
    'S-24-05-08-PM': [3353, 5, 6, 3],
    'S-24-04-02': [3574, 6, 3, 3],
    'S-24-05-10-PM': [3721, 2, 6, 3],
    'S-24-05-09-PM': [3815, 10, 5, 3],
    'S-24-01-12-PM': [4091, 7, 4, 3],
    'S-24-04-12': [4183, 4, 2, 3],
    'S-24-04-03': [4460, 7, 3, 3],
    'S-24-02-07-AM': [4493, 7, 3, 3],
    'S-24-02-02': [4787, 12, 3, 3],
    'S-24-01-24': [5047, 4, 4, 3],
    'S-24-01-23': [5080, 7, 4, 3],
    'S-24-01-30-PM': [5381, 14, 3, 3+1/6],
    'S-24-01-17-PM': [5388, 4, 5, 3],
    'S-24-02-06-PM': [5394, 15, 0, 3],
    'S-24-05-24-AM': [5446, 12, 1, 3],
    'S-24-02-07-PM': [5491, 12, 0, 3],
    'S-24-01-25': [5667, 8, 2, 3],
    'S-24-04-23-PM': [5716, 17, 1, 3],
    'S-24-01-30-AM': [5772, 6, 0, 3],
    'S-24-04-25-PM': [5810, 15, 0, 3],
    'S-24-04-25-AM': [5823, 4, 0, 3],
    'S-24-02-06-AM': [5970, 8, 3, 3+1/3],
    'S-24-02-01': [5994, 9, 5, 3],
    'S-24-01-11': [6031, 13, 2, 3],
    'S-24-05-23-FML': [6105, 20, 1, 3],
    'S-24-05-23-AM': [6111, 13, 3, 3],
    'S-24-01-12-AM': [6216, 13, 1, 3],
    'S-24-01-16': [6302, 13, 0, 3],
    'S-24-01-17-AM': [6358, 8, 1, 3],
    'S-24-05-09-AM': [6372, 14, 7, 3],
    'S-24-01-31': [6467, 13, 2, 3],
    'S-24-01-10': [6586, 11, 0, 3],
    'S-24-04-23-AM': [6662, 14, 1, 3],
    'S-24-04-24-AM': [6902, 24, 0, 3],
    'S-24-05-07': [7069, 20, 0, 2+5/6],
    'S-24-05-23-PM': [7321, 14, 2, 3],
    'S-24-01-26': [7376, 13, 1, 3],
    'S-24-05-10-AM': [7713, 23, 1, 3],
    'S-24-05-22-PM': [8401, 20, 1, 3],
    'S-24-05-08-AM': [8648, 22, 0, 2+5/6],
    'S-24-04-11': [8704, 16, 1, 3],
    'S-24-05-22-AM': [9221, 26, 1, 3]
    }

densities = [i[0] for i in list(all_control_data.values())]



mean_singularities = []

for i in all_control_data:
    s = get_n_singularities(i)
    mean_s = smooth_singularity_time_series(s)
    mean_singularities.append(mean_s)

cut = [7, 11, 12, 13, 14, 15, 20, 26]
cat1 = []
cat1_len = 600
for i, data in enumerate(mean_singularities):
    if list(all_control_data)[i] == "S-24-04-11" or list(all_control_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>6000 and densities[i]<7000:
        if i in cut:
            continue
        else:
            plt.plot(mean_singularities[i][:580], c='black', alpha = 0.2)
            cat1.append(mean_singularities[i][:580])        

cat1 = np.array([signal[:cat1_len] for signal in cat1])
mean_cat5 = np.mean(cat1, axis = 0)
sd_cat5 = np.std(cat1, axis =0)

reds = ["#fcae91",
"#fb6a4a",
"#de2d26",
"#a50f15"]

greens = ["#bae4b3",
"#74c476",
"#31a354",
"#006d2c"]

blues= ["#6baed6",
"#3182bd",
"#08519c"]

fig, axs = plt.subplots(3, 1, figsize=(6, 11), sharex=True)  
control_times = np.arange(3, 3+580/(3600/frame_intervals), 1/(3600/frame_intervals))
for i, data in enumerate(datasets):
    start_time=datasets[data][1]
    times = np.arange(start_time, start_time+600/(3600/frame_intervals), 1/(3600/frame_intervals))
    s = get_n_singularities(data)
    mean_s = smooth_singularity_time_series(s[:600], since_birth=False)
    if datasets[data][0]==0.25:
        axs[0].plot(times, mean_s[:len(times)], linewidth = 2, color = reds[i])
        axs[0].plot(control_times, mean_cat5, c='black', linewidth = 2)
        # axs[0].plot(control_times, mean_cat5+sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
        # axs[0].plot(control_times, mean_cat5-sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
    if datasets[data][0]==0.5:
        axs[1].plot(times[:-10], mean_s[:len(times)-10], linewidth = 2, color = greens[i-4])
        axs[1].plot(control_times, mean_cat5, c='black', linewidth = 2)
        # axs[1].plot(control_times, mean_cat5+sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
        # axs[1].plot(control_times, mean_cat5-sd_cat5, c='black', linewidth = 1.5, linestyle = "--")

    if datasets[data][0]==1:
        axs[2].plot(times, mean_s[:len(times)], linewidth = 2, color = blues[i-8])
        axs[2].plot(control_times, mean_cat5, c='black', linewidth = 2)
        # axs[2].plot(control_times, mean_cat5+sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
        # axs[2].plot(control_times, mean_cat5-sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
        # axs[2].fill_between(control_times, mean_cat5+sd_cat5, mean_cat5-sd_cat5, alpha=0.1, color = 'black')
for i in range(3):
    axs[i].set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    # axs[i].set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70],fontsize = 13)
    axs[i].tick_params(axis='y', labelsize = 13)
    axs[i].tick_params(axis='x', labelsize = 13)
    axs[i].fill_between(control_times, mean_cat5+sd_cat5, mean_cat5-sd_cat5, alpha=0.2, color = 'black')
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()










