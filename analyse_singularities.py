import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd
from scipy.signal import savgol_filter
import glob
from tqdm import tqdm
from sklearn.cluster import DBSCAN


data = "S-24-01-12-PM"
frame_intervals = 15

total_frames = 900

positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")

# # Fill in missing frames
# missing_rows = pd.concat([fill_gaps(group) for _, group in positive.groupby('particle')], ignore_index=True)
# positive = pd.concat([positive, missing_rows], ignore_index=True).sort_values(by=['particle', 'frame'])
# positive.reset_index(drop=True, inplace=True)

# missing_rows = pd.concat([fill_gaps(group) for _, group in negative.groupby('particle')], ignore_index=True)
# negative = pd.concat([negative, missing_rows], ignore_index=True).sort_values(by=['particle', 'frame'])
# negative.reset_index(drop=True, inplace=True)




_, positive = create_tracking_colormap(positive)
_, negative = create_tracking_colormap(negative)

s = get_n_singularities(data)
mean_s = smooth_singularity_time_series(s[:600], since_birth=False)
plt.scatter(np.arange(600), s[:600], s=1, alpha=0.6, c='black')
plt.plot(mean_s)
plt.xlabel("Time after starvation (h)", fontsize = 12) 
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/frame_intervals)*(np.array(times)-3)
plt.xticks(frames, times)
plt.ylabel("Number of singularities", fontsize = 12)
plt.show()

# plt.figure(figsize = (10,5))
# s, mean_s = get_n_singularities(data)
# plt.scatter(np.arange(500), s[:500], s=10, alpha=0.6, c='black')
# times = [3, 3.5, 4, 4.5, 5]
# frames = (3600/15)*(np.array(times)-3)
# plt.axvline(50, linestyle = 'dotted', c='red')
# plt.axvline(190, linestyle = 'dotted', c='red')
# plt.axvline(320, linestyle = 'dotted', c='red')
# plt.xticks(frames, times)
# plt.ylabel("Number of singularities", fontsize = 12)
# plt.xlabel("Time after starvation (h)", fontsize = 12) 
# plt.show()

# Plot particle longevity
for i in positive['particle'].unique():
    plt.scatter(list(positive[positive['particle']==i]['frame']),np.ones(len(positive[positive['particle']==i]))*i, color = positive[positive['particle']==i]['color'].iloc[0], s = 0.5)
for j in negative['particle'].unique():
    plt.scatter(list(negative[negative['particle']==j]['frame']),np.ones(len(negative[negative['particle']==j]))*j+0.5, color = negative[negative['particle']==j]['color'].iloc[0], s = 0.5)
times = [3, 3.5, 4, 4.5, 5, 5.5]
frames = (3600/15)*(np.array(times)-3)
plt.xticks(frames, times)
plt.xlabel("Time after starvation (h)");
plt.ylabel("Singularity ID");
plt.show()

# Plot net charge
plt.scatter(np.arange(600), [len(positive[positive['frame']==frame])-len(negative[negative['frame']==frame]) for frame in range(600)], s=1)
times = [3, 3.5, 4, 4.5, 5, 5.5, 6]
frames = (3600/15)*(np.array(times)-3)
plt.xticks(frames, times)
plt.xlabel("Time after starvation (h)");
plt.ylabel("Net Charge");
plt.show()

start_times = []
for i in positive['particle'].unique():
    start_times.append(list(positive[positive['particle']==i]['frame'])[0])
start_times = np.array(start_times)
times = [3, 3.5, 4, 4.5, 5, 5.5, 6]
frames = (3600/15)*(np.array(times)-3)
plt.hist(start_times[start_times<500]);
plt.xticks(frames, times);
plt.xlabel("Birth time after starvation (h)", fontsize = 12);
plt.ylabel("Frequency", fontsize = 12)
plt.show()

plt.figure(figsize = (8, 10))
# View single particle tracks
for particle_id in positive['particle'].unique():
    if len(positive[positive['particle']==particle_id]['x']) >1:
        print(f"Particle ID: {particle_id}")
        plt.scatter(positive[positive['particle']==particle_id]['x'],positive[positive['particle']==particle_id]['y'], 
                    c = positive[positive['particle']==particle_id]['frame'], s=len(positive[positive['particle']==particle_id]['x']/10), marker = '<')
for particle_id in negative['particle'].unique():
    if len(negative[negative['particle']==particle_id]['x']) >1:
        print(f"Particle ID: {particle_id}")
        plt.scatter(negative[negative['particle']==particle_id]['x'],negative[negative['particle']==particle_id]['y'], 
                    c = negative[negative['particle']==particle_id]['frame'], s=len(negative[negative['particle']==particle_id]['x']/10), marker = '>')
plt.ylim([276, 0])
plt.xlim([0, 210])
plt.colorbar(label = 'Frame number')
plt.show()


# View birth of singularities
plt.figure(figsize = (8, 10))
for particle_id in positive['particle'].unique():
    plt.scatter(list(positive[positive['particle']==particle_id]['x'])[0], list(positive[positive['particle']==particle_id]['y'])[0], s = len(positive[positive['particle']==particle_id]['x']))
for particle_id in negative['particle'].unique():
    plt.scatter(list(negative[negative['particle']==particle_id]['x'])[0], list(negative[negative['particle']==particle_id]['y'])[0], s = len(negative[negative['particle']==particle_id]['x']), marker = 's')
plt.ylim([276, 0])
plt.xlim([0, 210])
plt.show()

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

# View death of singularities
pairs = []
distances = []
plt.figure(figsize = (8, 10))
for particle_id in positive['particle'].unique():
    if list(positive[positive['particle']==particle_id]['frame'])[-1] == list(positive['frame'])[-1]:
        continue
    elif len(positive[positive['particle']==particle_id]) > 25:
        pairs.append([list(positive[positive['particle']==particle_id]['frame'])[-1], list(positive[positive['particle']==particle_id]['x'])[-1], list(positive[positive['particle']==particle_id]['y'])[-1]])
        plt.scatter(list(positive[positive['particle']==particle_id]['x'])[-1], list(positive[positive['particle']==particle_id]['y'])[-1], s = len(positive[positive['particle']==particle_id]['x']))
for particle_id in negative['particle'].unique():
    if list(negative[negative['particle']==particle_id]['frame'])[-1] == list(negative['frame'])[-1]:
        continue
    elif len(negative[negative['particle']==particle_id]) > 25:
        pairs.append([list(negative[negative['particle']==particle_id]['frame'])[-1], list(negative[negative['particle']==particle_id]['x'])[-1], list(negative[negative['particle']==particle_id]['y'])[-1]])
        plt.scatter(list(negative[negative['particle']==particle_id]['x'])[-1], list(negative[negative['particle']==particle_id]['y'])[-1], s = len(negative[negative['particle']==particle_id]['x']), marker = 's')
plt.ylim([276, 0])
plt.xlim([0, 210])
plt.show()
threshold_provided = 25
# Using the same function to find pairs below the threshold
pairs_below_threshold_provided = []
for pair in itertools.combinations(pairs, 2):
    distance = euclidean_distance(pair[0][1:], pair[1][1:]) 
    if distance < threshold_provided and pair[0][0]==pair[1][0]:
        pairs_below_threshold_provided.append(pair)
        distances.append(distance)



plt.figure(figsize = (8, 10))
# View all particle tracks
for particle_id in positive['particle'].unique():
    plt.plot(positive[positive['particle']==particle_id]['x'],positive[positive['particle']==particle_id]['y'])
for particle_id in negative['particle'].unique():
    plt.plot(negative[negative['particle']==particle_id]['x'],negative[negative['particle']==particle_id]['y'])
plt.ylim([276, 0])
plt.xlim([0, 210])
plt.show()


# for particle_id in positive['particle'].unique():
#     if len(positive[positive['particle']==particle_id]['period'])>100:
#         print(list(positive[positive['particle']==particle_id]['x'])[-1], list(positive[positive['particle']==particle_id]['y'])[-1])
#         plt.plot(positive[positive['particle']==particle_id]['period'])
#         plt.show()



creation = []
deaths = []
for i in positive['particle'].unique():
    start_times = list(positive[positive['particle']==i]['frame'])[0]
    creation.append(start_times)
    end_times = list(positive[positive['particle']==i]['frame'])[-1]
    deaths.append(end_times)
for i in negative['particle'].unique():
    start_times = list(negative[negative['particle']==i]['frame'])[0]
    creation.append(start_times)
    end_times = list(negative[negative['particle']==i]['frame'])[-1]
    deaths.append(end_times)

window = 40
creation_rates = []
death_rates = []
for frame in range(max(positive['frame'].unique())-window):
    window_range = set(range(frame, frame + window))

    count1 = sum(1 for number in creation if number in window_range)
    count2 = sum(1 for number in deaths if number in window_range)
    # Append the count to the list
    creation_rates.append(count1/(window*frame_intervals/60))
    death_rates.append(count2/(window*frame_intervals/60))

plt.plot(creation_rates, label = "Formation rate", linewidth = 2)
plt.plot(death_rates, label = 'Pruning rate', linewidth = 2)
plt.plot([creation_rates[i]-death_rates[i]for i in range(len(creation_rates))], label = "Net change", linewidth = 2)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/frame_intervals)*(np.array(times)-3)
plt.xticks(frames, times)
plt.xlabel("Time after starvation (h)")
plt.ylabel("Rate (events/min)")
plt.legend()




label_distribution = np.load(f"Data/{data}/Analysed_data/correlation_areas.npy")

plt.plot(np.arange(len(label_distribution)), label_distribution)
times = [3, 3.5, 4, 4.5, 5, 5.5]
frames = (3600/frame_intervals)*(np.array(times)-3)
plt.xlabel("Time after starvation (h)")
plt.xticks(frames, times);
plt.ylabel("Proportion of cells correlated")
plt.yscale('log')





fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Creating two subplots with shared x-axis

# Plot for the first subplot
axs[0].plot(creation_rates, label="Formation rate", linewidth=2)
axs[0].plot(death_rates, label='Pruning rate', linewidth=2)
axs[0].plot([creation_rates[i]-death_rates[i] for i in range(len(creation_rates))], label="Net change", linewidth=2)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/frame_intervals)*(np.array(times)-3)
axs[0].set_xticks(frames)
axs[0].set_xticklabels(times)
axs[0].set_ylabel("Rate (events/min)")
axs[0].legend()

# Plot for the second subplot
axs[1].plot(np.arange(len(label_distribution))[:600], label_distribution[:600])
axs[1].set_xlabel("Time after starvation (h)")
axs[1].set_xticks(frames)
axs[1].set_xticklabels(times)
axs[1].set_ylabel("Proportion of cells correlated")
axs[1].set_yscale('log')

plt.tight_layout()
plt.show()