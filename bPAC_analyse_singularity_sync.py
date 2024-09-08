import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd
from scipy.signal import savgol_filter
import glob
from tqdm import tqdm


datasets = {
    "B-24-04-19-PM": "Control",
    "B-24-05-02": "Control",
    "B-24-08-28": "Control",
    "B-24-04-12": 3,
    "B-24-04-19-AM": 3,
    "B-24-04-30": 3.5,
    "B-24-05-01": 3.5,
    "B-24-05-15": 3.5,
    "B-24-05-16": 3.5,
    "B-24-04-11": 4,
    "B-24-04-16": 4,
    "B-24-08-27": 4,
    "B-24-04-18-AM": 5, 
    "B-24-08-29-AM": 5,
    "B-24-04-17-AM": 6,
    }
frame_intervals = 20


# Define a list of colors to cycle through (you can change these colors to ones you prefer)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


for i, data in enumerate(datasets):
    
    singularities = pd.read_pickle(f"Data/{data}/Analysed_data/singularities.pkl")
    positive = singularities[singularities['spin']=='+']
    negative = singularities[singularities['spin']=='-']

    s = get_n_singularities(data)
    mean_s = smooth_singularity_time_series(s[:600], since_birth=False)
    if datasets[data] == 3:
        start_time = 2.5
    else:
        start_time=3

    color = colors[i % len(colors)]
    if data != "B-24-04-19-PM" and data!="B-24-05-02":
        times = np.arange(start_time, start_time+599/(3600/frame_intervals), 1/(3600/frame_intervals))
        plt.scatter(times, s[:len(times)], s=1, alpha=0.5, label = f"{datasets[data]}h", color = color)
        plt.axvline(datasets[data], linestyle = '--', color = color)
        
    else:
        times = np.arange(start_time, start_time+500/(3600/frame_intervals), 1/(3600/frame_intervals))
        plt.scatter(times, s[:len(times)], s=1, alpha=0.5, label = f"{datasets[data]}", color = color)
    plt.plot(times[:-40], mean_s[:len(times)-40], alpha = 0.8, linewidth = 4, color = color)
plt.xlabel("Time after starvation (h)", fontsize = 12) 
plt.ylabel("Number of singularities", fontsize = 12)
plt.legend()


fig, axs = plt.subplots(4, 1, figsize=(4, 11), sharex=True)  
for i, data in enumerate(datasets):
    if datasets[data] == 3:
        start_time = 2.5
    else:
        start_time=3
    s = get_n_singularities(data)
    if data=="B-24-08-27" or data == "B-24-08-29-AM":
        s=np.concatenate((np.array([0]*int(15*60/frame_intervals)), s))
        start_time = 2.75
    times = np.arange(start_time, start_time+500/(3600/frame_intervals), 1/(3600/frame_intervals))
    if data == "B-24-05-02":
        mean_s = smooth_singularity_time_series(s[:470], since_birth=False)
    else:
        mean_s = smooth_singularity_time_series(s[:600], since_birth=False)
    if data=="B-24-04-12":
        axs[0].plot(times[:-20], mean_s[:len(times)-20], linewidth = 4)
        axs[0].axvline(3, c='black')
    if data == "B-24-04-19-AM":
        axs[0].plot(times, mean_s[:len(times)], linewidth = 4)
        axs[0].axvline(3, c='black')
    if datasets[data]==3.5:
        axs[1].plot(times, mean_s[:len(times)], linewidth = 4)
        axs[1].axvline(3.5, c='black')
    if datasets[data]==4:
        axs[2].plot(times, mean_s[:len(times)], linewidth = 4)
        axs[2].axvline(4, c='black')
    if datasets[data]==5:
        axs[3].plot(times, mean_s[:len(times)], linewidth = 4)
        axs[3].axvline(5, c='black')
    if data=="B-24-04-19-PM":
        for j in range(4):
            axs[j].plot(times[:-40], mean_s[:len(times)-40], alpha=0.3, c='black')
    if data=="B-24-05-02":
        for j in range(4):
            axs[j].plot(times[:-30], mean_s[:len(times)-30], alpha=0.3, c='black')
    if data == "B-24-08-28":
        for j in range(4):
            axs[j].plot(times[:-30], mean_s[:len(times)-30], alpha=0.3, c='black')
for i in range(4):
    axs[i].set_xticks([3, 3.5, 4, 4.5, 5, 5.5])
    axs[i].set_xticklabels([3, 3.5, 4, 4.5, 5, 5.5], fontsize = 13)
    axs[i].set_yticks([0, 10, 20, 30, 40])
    axs[i].set_yticklabels([0, 10, 20, 30, 40], fontsize = 13)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()
