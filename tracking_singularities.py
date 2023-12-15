import trackpy as tp
import pims
import numpy as np
import matplotlib.pyplot as plt

data = "S-23-12-06"

positives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")>0.9
negatives = np.load(f"Data/{data}/Analysed_data/winding_numbers.npy")<-0.9

# Search for 7 pixel wide particles
f_p = tp.batch(positives, 7, processes = 1)

# Link if particle if next time frame is wthin 10 particles away and allow particle to be remembered for 15 frames if disappeared
t_p = tp.link(f_p, 10, memory = 15)

# Filter particle trajectories that are only present for 3 frames
t1_p = tp.filter_stubs(t_p, 3)

f_n = tp.batch(negatives, 7, processes = 1)
t_n = tp.link(f_n, 10, memory = 15)
t1_n = tp.filter_stubs(t_n, 3)

plt.figure()
tp.plot_traj(t1_p)
tp.plot_traj(t1_n)

t1_p.to_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
t1_n.to_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")


# for frame in range(50, 100):
#     for particle in t1_n[t1_n['frame'] == frame]['particle']:
#         particle_data = t1_n[(t1_n['frame'] == frame) & (t1_n['particle'] == particle)]
#         plt.scatter(particle_data['x'], particle_data['y'], color=color_map[particle], marker = 's')
