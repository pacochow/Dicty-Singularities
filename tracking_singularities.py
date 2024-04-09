import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt
from src.analysis_utils import *

data = "Other"

positives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")>0.9
negatives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")<-0.9
length = positives.shape[1]
width = positives.shape[2]

# Search for 7 pixel wide particles
f_p = tp.batch(positives, 7, processes = 1)

# Link if particle if next time frame is wthin 10 particles away and allow particle to be remembered for 15 frames if disappeared
t_p = tp.link(f_p, 10, memory = 15)

# Filter particle trajectories that are only present for 5 frames
t1_p = tp.filter_stubs(t_p, 5)

f_n = tp.batch(negatives, 7, processes = 1)
t_n = tp.link(f_n, 10, memory = 15)
t1_n = tp.filter_stubs(t_n, 5)



# Drop columns
t1_p = t1_p.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis = 1)
t1_n = t1_n.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis = 1)

plt.figure()
tp.plot_traj(t1_p)
tp.plot_traj(t1_n)

start_times = []
for i in t1_p['particle'].unique():
    start_times.append(list(t1_p[t1_p['particle']==i]['frame'])[0])
for i in t1_n['particle'].unique():
    start_times.append(list(t1_n[t1_n['particle']==i]['frame'])[0])
start_times = np.array(start_times)
times = [3, 3.5, 4, 4.5, 5, 5.5, 6]
frames = (3600/15)*(np.array(times)-3)
plt.hist(start_times);
plt.xticks(frames, times);
plt.xlabel("Birth time after starvation (h)", fontsize = 12);
plt.ylabel("Frequency", fontsize = 12)

t1_p, t1_n = filter_singularities(t1_p, t1_n, [25, 15, 20, 10], [40, 40, 40, 40], 800, 800, 800)

plt.figure()
tp.plot_traj(t1_p)
tp.plot_traj(t1_n)

# Save singularities
t1_p.to_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
t1_n.to_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")


start_times = []
for i in t1_p['particle'].unique():
    start_times.append(list(t1_p[t1_p['particle']==i]['frame'])[0])
for i in t1_n['particle'].unique():
    start_times.append(list(t1_n[t1_n['particle']==i]['frame'])[0])
start_times = np.array(start_times)
times = [3, 3.5, 4, 4.5, 5, 5.5, 6]
frames = (3600/15)*(np.array(times)-3)
plt.hist(start_times);
plt.xticks(frames, times);
plt.xlabel("Birth time after starvation (h)", fontsize = 12);
plt.ylabel("Frequency", fontsize = 12)
