import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd
from scipy.signal import savgol_filter
import glob
from tqdm import tqdm


datasets = {
    "A-24-05-02": [363, 30, 23],
    # "A-24-05-15": [386, 36, 15.1],
    "A-24-05-16": [342, 36, 15],
    "A-24-05-17": [329, 36, 37]
    }



frame_intervals = 20
start_time = 3

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

control = "B-24-04-19-PM"

s = get_n_singularities(control)
mean_s = smooth_singularity_time_series(s[:500], since_birth=False)
fig=plt.figure(figsize = (9,5))
plt.scatter(np.arange(3, 3+500/(3600/20), 1/(3600/20)), s[:500], s=1, alpha=0.6, c='black')
plt.plot(np.arange(3, 3+500/(3600/20), 1/(3600/20)), mean_s, c='black', linewidth = 2, alpha = 0.5)
plt.plot(np.arange(start_time+650/(3600/frame_intervals), start_time+850/(3600/frame_intervals), 1/(3600/frame_intervals)), [8]*len(np.arange(start_time+650/(3600/frame_intervals), start_time+850/(3600/frame_intervals), 1/(3600/frame_intervals))), color = 'black', linewidth = 2)
for i, data in enumerate(datasets):
    
    singularities = pd.read_pickle(f"Data/{data}/Analysed_data/singularities.pkl")
    positive = singularities[singularities['spin']=='+']
    negative = singularities[singularities['spin']=='-']

    s = get_n_singularities(data).astype(float)

    # Set number of singularities during photoactivation to nan
    s[datasets[data][0]-2:datasets[data][0]+datasets[data][1]-2] = [np.nan]*datasets[data][1]
    mean_s = smooth_singularity_time_series(s[:600], since_birth=False)

    color = colors[i % len(colors)]
    times = np.arange(start_time, start_time+(singularities['frame'].max()-10)/(3600/frame_intervals), 1/(3600/frame_intervals))
    plt.scatter(times, s[:len(times)], s=1, alpha=0.3, color = color)
    plt.plot(times[:-40], mean_s[:len(times)-40], alpha = 0.8, linewidth = 4, color = color, label = f"{data}")
    plt.scatter(np.arange(start_time+650/(3600/frame_intervals), start_time+850/(3600/frame_intervals), 1/(3600/frame_intervals)), [datasets[data][2]]*len(np.arange(start_time+650/(3600/frame_intervals), start_time+850/(3600/frame_intervals), 1/(3600/frame_intervals))), color = color, s=1)
# plt.axhline(11, linestyle = '--', color = 'black', alpha = 0.5, label = 'Predicted')
# plt.xlabel("Time after starvation (h)", fontsize = 12) 
# plt.ylabel("Number of singularities", fontsize = 12)
plt.axvline(5, c='black', linestyle = '--')
plt.xticks([3, 4, 5, 6, 7], fontsize = 13)
plt.yticks([0, 10, 20, 30, 40], fontsize = 13)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.legend()
plt.show()



