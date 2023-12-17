import numpy as np
from src.animate import *
from src.analysis_utils import *
import pandas as pd

data = "S-23-12-06"
frame_intervals = 15

total_frames = 900

positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")

# Number of singularities over time
n_singularities = [len(positive[positive['frame']==i])+len(negative[negative['frame']==i]) for i in range(total_frames)]
plt.scatter(np.arange(len(n_singularities)), n_singularities, s=1)
plt.xlabel("Frame")
plt.ylabel("Number of singularities")
plt.show()

# View single particle tracks
for particle_id in range(positive['particle'].max()):
    if len(positive[positive['particle']==particle_id]['x']) >200:
        print(f"Particle ID: {particle_id}")
        plt.scatter(positive[positive['particle']==particle_id]['x'],positive[positive['particle']==particle_id]['y'], 
                    c = positive[positive['particle']==particle_id]['frame'], s=5)
        plt.xlim([0, 276])
        plt.ylim([210, 0])
        plt.colorbar()
        plt.show()

# View all particle tracks
# for particle_id in range(negative['particle'].max()):
    
#     plt.plot(negative[negative['particle']==particle_id]['x'],negative[negative['particle']==particle_id]['y'])
#     plt.xlim([0, 276])
#     plt.ylim([210, 0])
