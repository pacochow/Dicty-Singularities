import numpy as np
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import sem, linregress 
from scipy.spatial import KDTree
import pingouin as pg
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.stats import norm

frame_intervals = 15

all_data = {
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

densities = [i[0] for i in list(all_data.values())]
aggregates = [i[1] for i in list(all_data.values())]
late_prune = [i[2] for i in list(all_data.values())]
start_times = [i[3] for i in list(all_data.values())]

#### SINGULARITY TIME PLOTS
singularities = []
mean_singularities = []
peaks = []
peak_frames = []
peak_times = []
final_number = []
for i in all_data:
    s = get_n_singularities(i)
 
    mean_s = smooth_singularity_time_series(s)

    singularities.append(s)
    mean_singularities.append(mean_s)
    peaks.append(np.max(s))
    if i == "S-24-04-25-PM" or i=="S-24-04-11":
        peak_frames.append(np.nan)
    else:
        peak_frames.append(np.argmax(s))
    peak_times.append(all_data[i][3]*60 + np.argmax(s)/4)

    

colormap = plt.cm.viridis  
num_plots = len(all_data)

# Generating colors from the colormap
color_indices = np.linspace(0, 1, num_plots)
colors = [colormap(i) for i in color_indices]

plt.figure(figsize = (10,8))
for i, data in enumerate(singularities):
    final_number.append(data[np.array(data)!=0][-1])
    start_time =start_times[i]
    times = np.arange(start_time, start_time+500/(3600/frame_intervals), 1/(3600/frame_intervals))
    plt.scatter(times, data[:len(times)], s=1, alpha=0.6, c=[colors[i]])


for i, data in enumerate(mean_singularities):
    start_time = start_times[i]
    times = np.arange(start_time, start_time+500/(3600/frame_intervals), 1/(3600/frame_intervals))

    plt.plot(times, data[:len(times)], c = colors[i], label = densities[i])

plt.xlabel("Time after starvation (h)", fontsize = 12)
plt.ylabel("Number of singularities", fontsize = 12)
plt.legend(loc='upper right', fontsize = 7)
plt.show()


# Plot time since first singularity formed

mean_singularities = []
for i, s in enumerate(singularities):

    mean_s = smooth_singularity_time_series(s, since_birth=True)
    mean_singularities.append(mean_s)

# FIG 2F
fig, ax = plt.subplots(figsize=(10, 5))  
cut = [7, 11, 12, 13, 14, 15, 20, 26]
for i in range(len(all_data)):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    elif i in cut:
        ax.plot(mean_singularities[i][:380], c=colors[i])
    else:
        ax.plot(mean_singularities[i][:480], c=colors[i])

# ax.set_xlabel("Time since first singularity formed (h)", fontsize=12)
times = [0, 0.5, 1, 1.5, 2]
frames = (3600/15)*(np.array(times))
ax.set_xticks(frames)
ax.set_xticklabels(times, fontsize = 13)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_yticklabels([0, 10, 20, 30, 40, 50], fontsize = 13)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# ax.set_ylabel("Number of singularities", fontsize=12)
plt.show()
# sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', ticks = [0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1])
# cbar.set_ticklabels([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], fontsize =14)  
# cbar.ax.xaxis.set_ticks_position('top')
# labels = densities
# plt.show()


# FIG 2G
fig, axs = plt.subplots(6, 1, figsize=(4, 15), sharex=True)  
c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>2000 and densities[i]<3000:
        if i in cut:
            axs[0].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[0].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat1 = np.mean(c_cat1, axis = 0)
sd_cat1 = np.std(c_cat1, axis =0)
axs[0].plot(mean_cat1, c='black', linewidth = 2)
# axs[0].plot(mean_cat1+sd_cat1, c='black', linewidth = 1.5, linestyle = "--")
# axs[0].plot(mean_cat1-sd_cat1, c='black', linewidth = 1.5, linestyle = "--")
axs[0].fill_between(np.arange(len(mean_cat1)), mean_cat1-sd_cat1, mean_cat1+sd_cat1, alpha=0.2,color='black')


c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>3000 and densities[i]<4000:
        if i in cut:
            axs[1].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[1].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat2 = np.mean(c_cat1, axis = 0)
sd_cat2 = np.std(c_cat1, axis =0)
axs[1].plot(mean_cat2, c='black', linewidth = 2)
# axs[1].plot(mean_cat2+sd_cat2, c='black', linewidth = 1.5, linestyle = "--")
# axs[1].plot(mean_cat2-sd_cat2, c='black', linewidth = 1.5, linestyle = "--")
axs[1].fill_between(np.arange(len(mean_cat2)), mean_cat2-sd_cat2, mean_cat2+sd_cat2, alpha=0.2,color='black')


c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>4000 and densities[i]<5000:
        if i in cut:
            axs[2].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[2].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat3 = np.mean(c_cat1, axis = 0)
sd_cat3 = np.std(c_cat1, axis =0)
axs[2].plot(mean_cat3, c='black', linewidth = 2)
# axs[2].plot(mean_cat3+sd_cat3, c='black', linewidth = 1.5, linestyle = "--")
# axs[2].plot(mean_cat3-sd_cat3, c='black', linewidth = 1.5, linestyle = "--")
axs[2].fill_between(np.arange(len(mean_cat3)), mean_cat3-sd_cat3, mean_cat3+sd_cat3, alpha=0.2,color='black')


c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>5000 and densities[i]<6000 :
        if i in cut:
            axs[3].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[3].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat4 = np.mean(c_cat1, axis = 0)
sd_cat4 = np.std(c_cat1, axis =0)
axs[3].plot(mean_cat4, c='black', linewidth = 2)
# axs[3].plot(mean_cat4+sd_cat4, c='black', linewidth = 1.5, linestyle = "--")
# axs[3].plot(mean_cat4-sd_cat4, c='black', linewidth = 1.5, linestyle = "--")
axs[3].fill_between(np.arange(len(mean_cat4)), mean_cat4-sd_cat4, mean_cat4+sd_cat4, alpha=0.2,color='black')

c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>6000 and densities[i]<7000:
        if i in cut:
            axs[4].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[4].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat5 = np.mean(c_cat1, axis = 0)
sd_cat5 = np.std(c_cat1, axis =0)
axs[4].plot(mean_cat5, c='black', linewidth = 2)
# axs[4].plot(mean_cat5+sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
# axs[4].plot(mean_cat5-sd_cat5, c='black', linewidth = 1.5, linestyle = "--")
axs[4].fill_between(np.arange(len(mean_cat5)), mean_cat5-sd_cat5, mean_cat5+sd_cat5, alpha=0.2,color='black')

c_cat1 = []
cat1_len = 480
for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    if densities[i]>7000 and densities[i]<8000:
        if i in cut:
            axs[5].plot(mean_singularities[i][:380], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:380])
            cat1_len = 380
        else:
            axs[5].plot(mean_singularities[i][:480], c='black', alpha = 0.2)
            c_cat1.append(mean_singularities[i][:480])        

c_cat1 = np.array([signal[:cat1_len] for signal in c_cat1])
mean_cat6 = np.mean(c_cat1, axis = 0)
sd_cat6 = np.std(c_cat1, axis =0)
axs[5].plot(mean_cat6, c='black', linewidth = 2)
# axs[5].plot(mean_cat6+sd_cat6, c='black', linewidth = 1.5, linestyle = "--")
# axs[5].plot(mean_cat6-sd_cat6, c='black', linewidth = 1.5, linestyle = "--")
axs[5].fill_between(np.arange(len(mean_cat6)), mean_cat6-sd_cat6, mean_cat6+sd_cat6, alpha=0.2,color='black')

# axs[2].set_xlabel("Time since first singularity formed (h)", fontsize=12)
times = [0, 0.5, 1, 1.5, 2]
frames = (3600/15)*(np.array(times))
axs[5].set_xticks(frames)
axs[5].set_xticklabels(times, fontsize = 14)
for i in range(6):
    axs[i].set_yticks([0, 20, 40, 60])
    axs[i].set_yticklabels([0, 20, 40, 60], fontsize = 14)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()


fig, ax=  plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(10)
interval = num_plots//6
ax.plot(mean_cat1, linewidth = 2,c=colors[interval])
ax.fill_between(np.arange(len(mean_cat1)), mean_cat1-sd_cat1, mean_cat1+sd_cat1, alpha=0.1,color=colors[interval])
ax.plot(mean_cat2, linewidth = 2, c= colors[interval*2])
ax.fill_between(np.arange(len(mean_cat2)), mean_cat2-sd_cat2, mean_cat2+sd_cat2, alpha=0.1,color=colors[interval*2])
ax.plot(mean_cat3, linewidth = 2, c=colors[interval*3])
ax.fill_between(np.arange(len(mean_cat3)), mean_cat3-sd_cat3, mean_cat3+sd_cat3, alpha=0.1,color=colors[interval*3])
ax.plot(mean_cat4, linewidth = 2, c=colors[interval*4])
ax.fill_between(np.arange(len(mean_cat4)), mean_cat4-sd_cat4, mean_cat4+sd_cat4, alpha=0.1,color=colors[interval*4])
ax.plot(mean_cat5, linewidth = 2, c=colors[interval*5])
ax.fill_between(np.arange(len(mean_cat5)), mean_cat5-sd_cat5, mean_cat5+sd_cat5, alpha=0.1,color=colors[interval*5])
ax.plot(mean_cat6, linewidth = 2, c=colors[interval*6])
ax.fill_between(np.arange(len(mean_cat6)), mean_cat6-sd_cat6, mean_cat6+sd_cat6, alpha=0.1,color=colors[interval*6])
times = [0, 0.5, 1, 1.5, 2]
frames = (3600/15)*(np.array(times))
ax.set_xticks(frames);
ax.set_xticklabels(times);
plt.show()




plt.scatter(densities, peaks, c='black', s=10)
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Maximum number of singularities");
z = np.polyfit(densities, peaks, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, peaks)
print(r_value**2, p_value)



# CREATION

# Plot histogram of singularity creation times
all_start_frames = []
earliest_start_frames = []
relative_start_frames = []
super_sing_starts = []
singularities_to_be_pruned = []
stabilising_frames = []
singularity_creation_rate = []
singularity_death_rate = []
singularity_creation = []
singularity_death = []
earliest_start_times = []
creations_post_stabilise = []
longevities = []
for j, date in enumerate(all_data):
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    
    earliest_start_frame = singularities['frame'].min()
    if date=="S-24-04-25-PM" or date=="S-24-04-11":
        earliest_start_frames.append(np.nan)
    else:
        earliest_start_frames.append(earliest_start_frame)
    earliest_start_times.append(all_data[date][3]*60+earliest_start_frame/4) # in minutes
    singularity_start_frames = []
    singularity_death_frames = []
    longevity = []
    creation_post_stabilise = 0
    for i in singularities['particle'].unique():
        singularity_start_frames.append(list(singularities[singularities['particle']==i]['frame'])[0])
        if date!="S-24-04-11" and date!= "S-24-04-25-PM":
            all_start_frames.append(list(singularities[singularities['particle']==i]['frame'])[0])
        relative_start_frames.append(list(singularities[singularities['particle']==i]['frame'])[0]-earliest_start_frame)
        if list(singularities[singularities['particle']==i]['frame'])[-1] ==max(singularities['frame']) and date!="S-24-04-11" and date!= "S-24-04-25-PM":
            super_sing_starts.append(list(singularities[singularities['particle']==i]['frame'])[0])
        longevity.append(len(singularities[singularities['particle']==i]))
    # stabilising_frame = int(np.percentile(singularity_start_frames, 95))+20
    x=np.sort(singularity_start_frames)
    x=x.reshape(len(x), 1)
    clustering = DBSCAN(eps = 15, min_samples = 10).fit(x)
    # If no outliers
    if np.diff(clustering.labels_[-20:]).sum()==0:
        stabilising_frame = x.max()
    else:
        # If outliers, then take last non-outlier creation frame as stabilising frame
        stabilising_frame = x[-(np.diff(clustering.labels_)!=0)[::-1].argmax()-2][0]
    for i in singularities['particle'].unique():
        if list(singularities[singularities['particle']==i]['frame'])[-1]<stabilising_frame:
            singularity_death_frames.append(list(singularities[singularities['particle']==i]['frame'])[-1])
        if list(singularities[singularities['particle']==i]['frame'])[0]>stabilising_frame:
            creation_post_stabilise+=1
    singularities_to_be_pruned.append(len(singularities[singularities['frame']==stabilising_frame]))
    # if date == "S-24-04-11" or date=="S-24-04-25-PM":
    #     stabilising_frames.append(np.nan)
    # else:
    stabilising_frames.append(stabilising_frame)
    singularity_creation.append(len(np.array(singularity_start_frames)<stabilising_frame))
    singularity_death.append(len(np.array(singularity_death_frames)<stabilising_frame))
    singularity_creation_rate.append(len(np.array(singularity_start_frames)<stabilising_frame)/((stabilising_frame-np.min(singularity_start_frames))/4))
    singularity_death_rate.append(len(np.array(singularity_death_frames)<stabilising_frame)/((stabilising_frame-np.min(singularity_start_frames))/4))
    creations_post_stabilise.append(creation_post_stabilise)
    longevities.append(longevity)
relative_start_frames = np.array(relative_start_frames)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/15)*(np.array(times)-3)

longevity_means=[]
for i in longevities:
    
    longevity_means.append(np.mean(i)/4)
plt.figure(figsize = (7,6))
plt.scatter(densities, longevity_means, color = 'black')
plt.show()

fig=plt.figure(figsize = (10,2))
plt.hist(all_start_frames, density = True, bins = 30, label = "All singularities");
plt.hist(super_sing_starts, density = True, bins = 30, color='red', alpha =0.5, label = 'Supersingularities')
plt.xticks(frames, times, fontsize = 13);
plt.yticks([0, 0.007], fontsize = 13)
# plt.xlabel("Birth time after starvation (h)", fontsize = 12);
# plt.ylabel("Probability density", fontsize = 12)
# plt.legend()
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()


# FIG 2J
fig, axs = plt.subplots(2, 1, figsize=(4, 10), sharex=True)  
axs[0].scatter(densities, singularity_creation, c='tab:blue')
axs[0].scatter(densities, singularity_death, c='tab:red')

axs[0].set_yticks([100, 150, 200, 250, 300])
axs[0].set_yticklabels([100, 150, 200, 250, 300], fontsize = 14)
axs[1].scatter(densities, singularity_creation_rate, c='tab:blue', label = "Birth")
axs[1].scatter(densities, singularity_death_rate, c='tab:red', label = "Death")
axs[1].scatter(densities, np.array(singularity_creation_rate)-np.array(singularity_death_rate), c='tab:green')
# axs[1].ylim([-1, 10])
axs[1].set_xticks([2000, 4000, 6000, 8000, 10000])
axs[1].set_xticklabels([2000, 4000, 6000, 8000, 10000], fontsize = 14)
axs[1].set_yticks([0, 2, 4, 6, 8, 10])
axs[1].set_xlim([1500, 10000])
axs[1].set_yticklabels([0, 2, 4, 6, 8, 10], fontsize = 14)
# plt.xlabel("Initial cell density ($cells/mm^2$)");
# plt.ylabel("Rate (events/min)")
# plt.legend()
for i in range(2):
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()

plt.figure(figsize = (7,6))
plt.scatter(densities, np.array(singularity_creation_rate)-np.array(singularity_death_rate), c='black')
plt.show()

plt.scatter(densities, earliest_start_times)
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Time of first singularity birth (mins)")
plt.show()

#### SINGULARITY PERIODS

end_periods_sd = []
end_periods_mean = []
for date in all_data:
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    
    end_period = []
    end_frame = np.max(list(singularities['frame']))
    for particle_id in singularities['particle'].unique():
        if list(singularities[singularities['particle']==particle_id]['frame'])[-1] == end_frame:
            frames = list(singularities[singularities['particle']==particle_id]['frame'])[-50:-30]
            periods = list(singularities[(singularities['frame'].isin(frames)) & (singularities['particle'] == particle_id)]['angular period'])
            end_period.append(np.nanmean(periods))

    # Remove far outliers
    q1 = np.nanpercentile(end_period, 25)
    q3 = np.nanpercentile(end_period, 75)
    iqr = q3 - q1
    lower_bound = q1 - 5 * iqr
    upper_bound = q3 + 5 * iqr
    end_period = [x for x in end_period if x >= lower_bound and x <= upper_bound]
    end_periods_mean.append(np.nanmean(end_period))
    end_periods_sd.append(np.nanstd(end_period))

plt.scatter(densities, end_periods_sd, c = 'black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Standard deviation of periods of supersingularities")
z = np.polyfit(densities, end_periods_sd, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, end_periods_sd)
print(r_value**2, p_value)
plt.show()

periods_across_time_mean = []
periods_across_time_sd = []
for i, date in enumerate(all_data):
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    period_sd = []
    period_mean = []
    for frame in np.arange(stabilising_frames[i], singularities['frame'].max()):

        # Get periods of all singularities in that frame
        periods = list(singularities[singularities['frame']==frame]['angular period'])
        if np.sum(np.isnan(periods))==len(periods):
            period_mean.append(np.nan)
            period_sd.append(np.nan)
            continue
        else:
            period_mean.append(np.nanmean(periods))
            period_sd.append(np.nanstd(periods))
    periods_across_time_mean.append(period_mean)
    periods_across_time_sd.append(period_sd)


times = np.array([3, 4, 5, 6, 7])
fig, ax = plt.subplots(figsize=(10, 7))  
for i in range(len(all_data)):
    start_frame = list(all_data.values())[i][3]*60*60/15
    ax.plot(np.arange(start_frame+stabilising_frames[i]+len(periods_across_time_mean[i])), [np.nan]*int(start_frame+stabilising_frames[i])+periods_across_time_mean[i], c=colors[i])
ax.set_yticks([4, 6, 8, 10, 12, 14])
# ax.set_xlabel("Time since starvation (hr)")
ax.set_xticks(times*4*60)
ax.set_xticklabels([])
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(axis='y', labelsize = 13)
ax.tick_params(axis='x', labelsize = 13)
plt.xlim([600, 1700])

# ax.set_ylabel("Mean angular period of all singularities (mins)")
# sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])
# cbar.set_ticklabels([2000, 9000])  

# cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)
# plt.show(cbar)
plt.show()



# cats = np.full((3*60*4+900, len(control_data)), np.nan)
# for i in range(len(control_data)):
#     start_frame = list(control_data.values())[i][3]*60*60/15
#     cats[int(start_frame+stabilising_times[i]):int(start_frame+stabilising_times[i])+len(periods_across_time_mean[i]), i] = periods_across_time_mean[i]
# for i in range(len(control_data)):
#     if densities[i]>1000 and densities[i]<3000:
#         # cats1 = cats[:, :i]
#         # mask = ~np.isnan(cats1).any(axis=1)
#         # cat1 = np.where(mask, np.mean(cats1, axis=1), np.nan)
#         # std1 = np.where(mask, np.std(cats1, axis=1), np.nan)
#         cat1 = np.nanmean(cats[:, :i], axis=1)
#         std1 = np.nanstd(cats[:, :i], axis=1)
#         cat1[:1155]=[np.nan]*1155
#         cat1[1550:]=[np.nan]*len(cat1[1550:])
#         a = i
#     if densities[i]>3000 and densities[i]<4000:
#         # cats2 = cats[:, a:i]
#         # mask = ~np.isnan(cats2).any(axis=1)
#         # cat2 = np.where(mask, np.mean(cats2, axis=1), np.nan)
#         # std2 = np.where(mask, np.std(cats2, axis=1), np.nan)
#         cat2 = np.nanmean(cats[:, a:i], axis=1)
#         std2 = np.nanstd(cats[:, a:i], axis=1)
#         cat2[:1040] = [np.nan]*1040
#         b=i
#     if densities[i]>4000 and densities[i]<5000:
#         # cats3 = cats[:, b:i]
#         # mask = ~np.isnan(cats3).any(axis=1)
#         # cat3 = np.where(mask, np.mean(cats3, axis=1), np.nan)
#         # std3 = np.where(mask, np.std(cats3, axis=1), np.nan)
#         cat3 = np.nanmean(cats[:, b:i], axis=1)
#         std3 = np.nanstd(cats[:, b:i], axis=1)
#         cat3[:1020] = [np.nan]*1020
#         c=i
#     if densities[i]>5000 and densities[i]<6000:
#         # cats4 = cats[:, c:i]
#         # mask = ~np.isnan(cats4).any(axis=1)
#         # cat4 = np.where(mask, np.mean(cats4, axis=1), np.nan)
#         # std4 = np.where(mask, np.std(cats4, axis=1), np.nan)
#         cat4 = np.nanmean(cats[:, c:i], axis=1)
#         std4 = np.nanstd(cats[:, c:i], axis=1)
#         d=i
#     if densities[i]>6000 and densities[i]<7000:
#         # cats5 = cats[:, d:i]
#         # mask = ~np.isnan(cats5).any(axis=1)
#         # cat5 = np.where(mask, np.mean(cats5, axis=1), np.nan)
#         # std5 = np.where(mask, np.std(cats5, axis=1), np.nan)
#         cat5 = np.nanmean(cats[:, d:i], axis=1)
#         std5 = np.nanstd(cats[:, d:i], axis=1)
#         cat5[:920] = [np.nan]*920
#         e=i
#     if densities[i]>7000:
#         # cats6 = cats[:, e:i]
#         # mask = ~np.isnan(cats6).any(axis=1)
#         # cat6 = np.where(mask, np.mean(cats6, axis=1), np.nan)
#         # std6 = np.where(mask, np.std(cats6, axis=1), np.nan)
#         cat6 = np.nanmean(cats[:, e:i], axis=1)
#         std6 = np.nanstd(cats[:, e:i], axis=1)
#         cat6[:900] = [np.nan]*900
# plt.plot(cat1, c='tab:blue', linewidth = 2, label = "2000-3000")
# plt.plot(cat1+std1, c='tab:blue', linewidth = 1.5, linestyle = "--")
# plt.plot(cat1-std1, c='tab:blue', linewidth = 1.5, linestyle = "--")

# plt.plot(cat2, c='tab:orange', linewidth = 2, label = "3000-4000")
# plt.plot(cat2+std2, c='tab:orange', linewidth = 1.5, linestyle = "--")
# plt.plot(cat2-std2, c='tab:orange', linewidth = 1.5, linestyle = "--")

# plt.plot(cat3, c='tab:green', linewidth = 2, label = "4000-5000")
# plt.plot(cat3+std3, c='tab:green', linewidth = 1.5, linestyle = "--")
# plt.plot(cat3-std3, c='tab:green', linewidth = 1.5, linestyle = "--")

# plt.plot(cat4, c='tab:red', linewidth = 2, label = "5000-6000")
# plt.plot(cat4+std4, c='tab:red', linewidth = 1.5, linestyle = "--")
# plt.plot(cat4-std4, c='tab:red', linewidth = 1.5, linestyle = "--")

# plt.plot(cat5, c='tab:purple', linewidth = 2, label = "6000-7000")
# plt.plot(cat5+std5, c='tab:purple', linewidth = 1.5, linestyle = "--")
# plt.plot(cat5-std5, c='tab:purple', linewidth = 1.5, linestyle = "--")

# plt.plot(cat6, c='tab:pink', linewidth = 2, label = '>7000')
# plt.plot(cat6+std6, c='tab:pink', linewidth = 1.5, linestyle = "--")
# plt.plot(cat6-std6, c='tab:pink', linewidth = 1.5, linestyle = "--")
# plt.legend()
# plt.xticks(times*4*60, times)
# plt.show()



#### SINGULARITY SPEEDS

mean_super_speed = []
sd_super_speed = []
mean_all_speed = []
sd_all_speed = []

for date in all_data:
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")

    end_frame = np.max(list(singularities['frame']))
    super_speed = []
    all_speed = []
    for particle_id in singularities['particle'].unique():
        distance_travelled = 0
        xs = list(singularities[singularities['particle']==particle_id]['x'])
        ys = list(singularities[singularities['particle']==particle_id]['y'])
        for i in range(len(xs)-1):
            distance_travelled += np.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)
        if len(singularities[singularities['particle']==particle_id])>20:
            all_speed.append((distance_travelled*8*5.5/1000)/(len(singularities[singularities['particle']==particle_id])*15/60))
        if list(singularities[singularities['particle']==particle_id]['frame'])[-1] == end_frame:
            super_speed.append((distance_travelled*8*5.5/1000)/(len(singularities[singularities['particle']==particle_id])*15/60))

    mean_all_speed.append(np.mean(all_speed))
    mean_super_speed.append(np.mean(super_speed))
    sd_super_speed.append(sem(super_speed))
    sd_all_speed.append(sem(all_speed))

plt.errorbar(densities, mean_all_speed, yerr = sd_all_speed, fmt = 'o', ecolor='black')
plt.xlabel("Initial cell density ($cells/mm^2$)", size = 12)
plt.ylabel("Mean speed of all singularities (mm/min)")
z = np.polyfit(densities, mean_all_speed, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, mean_all_speed)
print(r_value**2, p_value)

#### DEATHS
all_n_pairs = []
super_speed = []
density_distances = []
all_premature = []
total_singularities = []
all_premature_death_times = []
all_pair_death_times = []
all_pair_crowd = []
for date in all_data:
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    singularities['death'] = 'Other'
    pairs = []
    density_distance = []
    premature = []
    premature_death_times = []
    pair_death_times = []
    pair_crowd = []
    

    n_singularities = len(singularities['particle'].unique())
    total_singularities.append(n_singularities)
    for particle_id in singularities['particle'].unique():

        # If singularity died because it made it to the end then ignore
        if list(singularities[singularities['particle']==particle_id]['frame'])[-1] == list(singularities['frame'])[-1]:
            singularities.loc[singularities['particle']==particle_id, 'death'] = "Aggregate"
            continue

        # Only look at singularities that stabilized for over 20 frames for pair annihilations
        elif len(singularities[singularities['particle']==particle_id]) > 20:
            pairs.append([list(singularities[singularities['particle']==particle_id]['frame'])[-1], particle_id, list(singularities[singularities['particle']==particle_id]['x'])[-1], list(singularities[singularities['particle']==particle_id]['y'])[-1]])
        
        # Premature deaths
        else: 
            premature.append(particle_id)
            premature_death_times.append(list(singularities[singularities['particle']==particle_id]['frame'])[-1])
            singularities.loc[singularities['particle']==particle_id, 'death'] = "Premature"
    threshold_provided = 25
    # Find pairs close to each other and died in same frame 
    pairs_below_threshold_provided = []
    for pair in itertools.combinations(pairs, 2):
        distance = euclidean_distance(pair[0][2:], pair[1][2:]) 
        if distance < threshold_provided and pair[0][0]==pair[1][0]:
            pairs_below_threshold_provided.append(pair)
            super_speed.append(distance)
            density_distance.append(distance)
            pair_death_times.append(pair[0][0])
            pair_crowd.append(len(singularities[singularities['frame']==pair[0][0]]))
            singularities.loc[singularities['particle']==pair[0][1], 'death'] = "Pair"
            singularities.loc[singularities['particle']==pair[1][1], 'death'] = "Pair"
    
    density_distances.append(np.mean(density_distance))
    all_n_pairs.append(len(density_distance))
    all_premature.append(len(premature))
    all_premature_death_times.append(premature_death_times)
    all_pair_death_times.append(pair_death_times)
    all_pair_crowd.append(pair_crowd)
    # print(singularities[singularities['death']=='Other'])
    # singularities.to_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
super_speed = np.array(super_speed)*8*5.5/1000
density_distances = np.array(density_distances)*8*5.5/1000

plt.hist(super_speed, bins=5, edgecolor = 'black')
plt.xlabel("Distance between singularities at pair annihilation (mm)")
plt.ylabel("Density")
plt.axvline(super_speed.mean(), linestyle = 'dotted', c='black')
plt.show()

plt.scatter(densities, density_distances, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Distance between singularities at \npair annihilation (mm)")
z = np.polyfit(densities, density_distances, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, density_distances)
print(r_value**2, p_value)

plt.scatter(densities, all_n_pairs, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of pair annihilation events")
plt.yticks([0, 5, 10, 15, 20])
z = np.polyfit(densities, all_n_pairs, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, all_n_pairs)
print(r_value**2, p_value)

plt.figure(figsize = (7,4))
plt.scatter(densities, np.array(all_n_pairs)/np.array(singularities_to_be_pruned))
plt.show()

plt.scatter(densities, all_premature, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of premature deaths")
z = np.polyfit(densities, all_premature, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, all_premature)
print(r_value**2, p_value)

plt.scatter(densities, [np.mean(i) for i in all_pair_crowd])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Mean number of singularities during pair annihilation events")
z = np.polyfit(densities, [np.mean(i) for i in all_pair_crowd], 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, [np.mean(i) for i in all_pair_crowd])
print(r_value**2, p_value)

#### MOTILITY-BASED PRUNING
plt.scatter(densities, late_prune, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of motility-induced deaths")
z = np.polyfit(densities, late_prune, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, late_prune)
print(r_value**2, p_value)

 
plt.figure(figsize = (7,4))
plt.scatter(densities, all_n_pairs, label = "Pair annihilations")
plt.scatter(densities, late_prune, label = "Motility-induced")
plt.yticks([0, 5, 10, 15, 20])
plt.xticks([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], [])
plt.show()

plt.figure(figsize = (7,4))
plt.scatter(densities, np.array(all_n_pairs)/np.array(singularities_to_be_pruned), c='black')
plt.ylim([0, 1])
plt.xticks([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], [])
plt.show()


plt.scatter(densities, final_number, label = "After premature and pair annihilations")
plt.scatter(densities, aggregates, alpha = 0.5, label = "Final aggregates")
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of singularities")
plt.yticks([5, 10, 15, 20])
plt.legend()
z = np.polyfit(densities, final_number, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, final_number)
print(r_value**2, p_value)



#### GROUP BY DENSITY
low_premature = []
low_pair = []
low_motility = []

mid_premature = []
mid_pair = []
mid_motility = []

high_premature = []
high_pair = []
high_motility = []

for i in range(len(densities)):
    # if list(densities())[i] < 5000:
    #     low_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     low_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     low_motility.append(list(late_prune)[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    # elif list(densities())[i] < 6000:
    #     mid_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     mid_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     mid_motility.append(list(late_prune)[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))

    # else:
    #     high_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     high_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))
    #     high_motility.append(list(late_prune)[i]/(all_premature[i]+all_n_pairs[i]+list(late_prune)[i]))

    if densities[i] < 4000:
        low_premature.append(all_premature[i])
        low_pair.append(all_n_pairs[i])
        low_motility.append(list(late_prune)[i])
    elif densities[i] < 6000:
        mid_premature.append(all_premature[i])
        mid_pair.append(all_n_pairs[i])
        mid_motility.append(list(late_prune)[i])

    else:
        high_premature.append(all_premature[i])
        high_pair.append(all_n_pairs[i])
        high_motility.append(list(late_prune)[i])


mean_low_premature, sd_low_premature = np.mean(low_premature), sem(low_premature)
mean_low_pair, sd_low_pair = np.mean(low_pair), sem(low_pair)
mean_low_motility, sd_low_motility = np.mean(low_motility), sem(low_motility)
mean_mid_premature, sd_mid_premature = np.mean(mid_premature), sem(mid_premature)
mean_mid_pair, sd_mid_pair = np.mean(mid_pair), sem(mid_pair)
mean_mid_motility, sd_mid_motility = np.mean(mid_motility), sem(mid_motility)
mean_high_premature, sd_high_premature = np.mean(high_premature), sem(high_premature)
mean_high_pair, sd_high_pair = np.mean(high_pair), sem(high_pair)
mean_high_motility, sd_high_motility = np.mean(high_motility), sem(high_motility)

low_cat = np.array([mean_low_premature, mean_low_pair, mean_low_motility])
low_cat/= low_cat.sum()
mid_cat = np.array([mean_mid_premature, mean_mid_pair, mean_mid_motility])
mid_cat/= mid_cat.sum()
high_cat = np.array([mean_high_premature, mean_high_pair, mean_high_motility])
high_cat/= high_cat.sum()

premature_cat=np.array([mean_low_premature, mean_mid_premature, mean_high_premature])
premature_cat/=premature_cat.sum()
sd_premature_cat = np.array([sd_low_premature, sd_mid_premature, sd_high_premature])/np.sum([[mean_low_premature, mean_mid_premature, mean_high_premature]])

pair_cat=np.array([mean_low_pair, mean_mid_pair, mean_high_pair])
pair_cat/=pair_cat.sum()
sd_pair_cat = np.array([sd_low_pair, sd_mid_pair, sd_high_pair])/np.sum([[mean_low_pair, mean_mid_pair, mean_high_pair]])


motility_cat=np.array([mean_low_motility, mean_mid_motility, mean_high_motility])
motility_cat /= motility_cat.sum()
sd_motility_cat = np.array([sd_low_motility, sd_mid_motility, sd_high_motility])/np.sum([[mean_low_motility, mean_mid_motility, mean_high_motility]])

# # Create a figure and a set of subplots
# fig, ax = plt.subplots()

# # Calculate the position of each bar
# index = np.arange(3)
# bar_width = 0.2
# opacity = 0.8

# rects1 = plt.bar(index - bar_width, [mean_low_premature, mean_low_pair, mean_low_motility], bar_width, alpha=opacity, label='Low')
# rects2 = plt.bar(index, [mean_mid_premature, mean_mid_pair, mean_mid_motility], bar_width, alpha=opacity, label='Mid')
# rects3 = plt.bar(index + bar_width, [mean_high_premature, mean_high_pair, mean_high_motility], bar_width, alpha=opacity, label='High')

# plt.xlabel('Type of death', size = 12)
# plt.ylabel('Mean proportion', size = 12)
# plt.xticks(index, ('Premature', 'Pair annihilation', 'Motility-based'))
# plt.legend()

# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()

index = np.arange(3)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, [premature_cat[0], pair_cat[0], motility_cat[0]], bar_width, alpha=opacity, label='Low')
rects2 = plt.bar(index, [premature_cat[1], pair_cat[1], motility_cat[1]], bar_width, alpha=opacity, label='Mid')
rects3 = plt.bar(index + bar_width, [premature_cat[2], pair_cat[2], motility_cat[2]], bar_width, alpha=opacity, label='High')


plt.xlabel('Type of death', size = 12)
plt.ylabel('Mean proportion', size = 12)
plt.xticks(index, ('Premature', 'Pair annihilation', 'Motility-based'))
plt.errorbar(index-bar_width, [premature_cat[0], pair_cat[0], motility_cat[0]], [sd_premature_cat[0], sd_pair_cat[0], sd_motility_cat[0]], fmt='o')
plt.errorbar(index, [premature_cat[1], pair_cat[1], motility_cat[1]], [sd_premature_cat[1], sd_pair_cat[1], sd_motility_cat[1]], fmt='o')
plt.errorbar(index+bar_width, [premature_cat[2], pair_cat[2], motility_cat[2]], [sd_premature_cat[2], sd_pair_cat[2], sd_motility_cat[2]], fmt='o')
plt.legend()

plt.tight_layout()
plt.show()



#### FORMATION AND PRUNING RATES
all_creation_plots = []
all_death_plots = []
for date in all_data:
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    
    creation = []
    deaths = []
    for i in singularities['particle'].unique():
        all_start_frames = list(singularities[singularities['particle']==i]['frame'])[0]
        creation.append(all_start_frames)
        end_times = list(singularities[singularities['particle']==i]['frame'])[-1]
        deaths.append(end_times)


    window = 40
    creation_rates = []
    death_rates = []
    for frame in range(max(singularities['frame'].unique())-window):
        window_range = set(range(frame, frame + window))

        count1 = sum(1 for number in creation if number in window_range)
        count2 = sum(1 for number in deaths if number in window_range)

        creation_rates.append(count1/(window*frame_intervals/60))
        death_rates.append(count2/(window*frame_intervals/60))
    all_creation_plots.append(creation_rates)
    all_death_plots.append(death_rates)

synchronisation_frame = []
creation_diffs = []
creation_peaks = []
prune_diffs = []
half_diffs = []
halfps = []
halfcs = []
for i, date in enumerate(all_data):
    
    creation_rates = all_creation_plots[i]
    death_rates = all_death_plots[i]

    label_distribution=np.load(f"Data/{date}/Analysed_data/correlation_areas.npy")
    if date == "S-24-04-25-PM" or date=="S-24-04-11":
        synchronisation_frame.append(np.nan)
        creation_peaks.append(np.nan)
    else:
        synchronisation_frame.append(np.argmax(label_distribution==label_distribution.max()))
        creation_peaks.append(np.argmax(creation_rates))

    

    
    creation_diffs.append(np.argmax(creation_rates)-synchronisation_frame[i])
    
    prune_diffs.append(np.argmax(death_rates)-synchronisation_frame[i])
    cs = []
    ps = []
    c=0
    p=0
    for t in range(len(creation_rates)):
        cs.append(c)
        c+=creation_rates[t]
        ps.append(p)
        p+=death_rates[t]
    halfc=np.abs(np.array(cs)-np.max(cs)/2).argmin()
    halfp=np.abs(np.array(ps)-np.max(ps)/2).argmin()
    half_diffs.append((halfp-halfc)/4)
    halfps.append(halfp)
    halfcs.append(halfc)

    # fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Creating two subplots with shared x-axis
    # print(densities[i], synchronisation_frame[i])
    # # Plot for the first subplot
    # axs[0].plot(creation_rates, label="Formation rate", linewidth=2)
    # axs[0].plot(death_rates, label='Pruning rate', linewidth=2)
    # axs[0].plot([creation_rates[i]-death_rates[i] for i in range(len(creation_rates))], label="Net change", linewidth=2)
    # axs[0].axvline(synchronisation_frame[i], linestyle = '--', color = 'black')
    # times = [3, 3.5, 4, 4.5, 5]
    # frames = (3600/frame_intervals)*(np.array(times)-3)
    # axs[0].set_xticks(frames)
    # axs[0].set_xticklabels(times)
    # axs[0].set_ylabel("Rate (events/min)")
    # axs[0].legend()

    # # Plot for the second subplot
    # axs[1].plot(np.arange(len(label_distribution)-200), label_distribution[:len(label_distribution)-200])
    # axs[1].axvline(synchronisation_frame[i], linestyle = '--', color = 'black')
    # axs[1].set_xlabel("Time after starvation (h)")
    # axs[1].set_xticks(frames)
    # axs[1].set_xticklabels(times)
    # axs[1].set_ylabel("Proportion of cells correlated")
    # axs[1].set_yscale('log')

    # plt.tight_layout()
    # plt.show()

times = np.array([3, 4, 5])
for i in [1, 18, 38]:
    plt.figure(figsize = (9,2))
    plt.plot(all_creation_plots[i][:500], c='tab:blue')
    plt.plot(all_death_plots[i][:500], c='tab:orange')
    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.axvline(halfcs[i], c = 'tab:blue', linestyle = '--')
    plt.axvline(halfps[i], c='tab:orange', linestyle = '--')
    if i==38:
        plt.xticks((times-3)*240, times)
    else:
        plt.xticks((times-3)*240, [])
    plt.show()


plt.figure(figsize = (7,6))
plt.scatter(densities, half_diffs, c='black')
plt.yticks([0, 2, 4, 6, 8, 10])
plt.show()

plt.scatter(densities, np.array(creation_diffs)*15/60, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Difference between time of peak creation rate \nand development (mins)")
plt.show()

plt.scatter(densities, np.array(peak_times)-np.array(earliest_start_times), c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Difference between time of peak singularities and \nfirst singularity creation (mins)")

longevity_means=[]
for i in longevities:
    
    longevity_means.append(np.mean(i)/4)



# FIG 2J
fig, axs = plt.subplots(2, 1, figsize=(4, 10), sharex=True)  
singularity_creation_range = [(stabilising_frames[i]-synchronisation_frame[i])*15/60 for i in range(len(all_data))]
axs[0].scatter(densities, singularity_creation_range, c='black')
axs[0].set_yticks([20, 30, 40, 50, 60, 70, 80])
axs[0].set_yticklabels([20, 30, 40, 50, 60, 70, 80], fontsize = 14)
# plt.xlabel("Initial cell density ($cells/mm^2$)");
# plt.ylabel("Difference between stabilisation \n and development time (mins)")
# z = np.polyfit(densities, singularity_creation_range, 1)
# p = np.poly1d(z)
# plt.plot(densities,p(densities),"r--", alpha =0.4)
axs[1].set_xticks([2000, 4000, 6000, 8000, 10000])
axs[1].set_xticklabels([2000, 4000, 6000, 8000, 10000], fontsize = 14)
axs[1].set_yticks([5, 10, 15, 20, 25])
axs[1].set_yticklabels([5, 10, 15, 20, 25], fontsize = 14)
# slope, intercept, r_value, p_value, std_err = linregress(densities, singularity_creation_range)
# print(r_value**2, p_value)
axs[1].set_xlim([1500, 10000])
axs[1].scatter(densities, longevity_means, color = 'black')
for i in range(2):
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
plt.tight_layout()
plt.show()


# FIG 2G
c_cat1=[]
c_cat2=[]
c_cat3=[]
c_cat4=[]
c_cat5=[]
c_cat6=[]

d_cat1=[]
d_cat2=[]
d_cat3=[]
d_cat4=[]
d_cat5=[]
d_cat6=[]
for i, date in enumerate(all_data):
    
    if date == "S-24-04-11" or date=="S-24-04-25-PM":
        continue
    earliest_singularity = earliest_start_frames[i]
    creation_rates = all_creation_plots[i][earliest_singularity:]
    death_rates = all_death_plots[i][earliest_singularity:]
    if all_data[date][0]>2000 and all_data[date][0]<3000:
        c_cat1.append(creation_rates)
        d_cat1.append(death_rates)
    if all_data[date][0]>3000 and all_data[date][0]<4000:
        c_cat2.append(creation_rates)
        d_cat2.append(death_rates)
    if all_data[date][0]>4000 and all_data[date][0]<5000:
        c_cat3.append(creation_rates)
        d_cat3.append(death_rates)
    if all_data[date][0]>5000 and all_data[date][0]<6000:
        c_cat4.append(creation_rates)
        d_cat4.append(death_rates)
    if all_data[date][0]>6000 and all_data[date][0]<7000:
        c_cat5.append(creation_rates)
        d_cat5.append(death_rates)
    if all_data[date][0]>7000 and all_data[date][0]<8000:
        c_cat6.append(creation_rates)
        d_cat6.append(death_rates)

creation_cat1_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat1])] for i in c_cat1]), axis=0)[:500]
creation_cat2_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat2])] for i in c_cat2]), axis=0)[:500]
creation_cat3_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat3])] for i in c_cat3]), axis=0)[:500]
creation_cat4_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat4])] for i in c_cat4]), axis=0)[:500]
creation_cat5_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat5])] for i in c_cat5]), axis=0)[:500]
creation_cat6_mean = np.mean(np.array([i[:np.min([len(i) for i in c_cat6])] for i in c_cat6]), axis=0)[:500]

creation_cat1_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat1])] for i in c_cat1]), axis=0)[:500]
creation_cat2_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat2])] for i in c_cat2]), axis=0)[:500]
creation_cat3_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat3])] for i in c_cat3]), axis=0)[:500]
creation_cat4_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat4])] for i in c_cat4]), axis=0)[:500]
creation_cat5_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat5])] for i in c_cat5]), axis=0)[:500]
creation_cat6_sd = np.std(np.array([i[:np.min([len(i) for i in c_cat6])] for i in c_cat6]), axis=0)[:500]

deaths_cat1_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat1])] for i in d_cat1]), axis=0)[:500]
deaths_cat2_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat2])] for i in d_cat2]), axis=0)[:500]
deaths_cat3_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat3])] for i in d_cat3]), axis=0)[:500]
deaths_cat4_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat4])] for i in d_cat4]), axis=0)[:500]
deaths_cat5_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat5])] for i in d_cat5]), axis=0)[:500]
deaths_cat6_mean = np.mean(np.array([i[:np.min([len(i) for i in d_cat6])] for i in d_cat6]), axis=0)[:500]

deaths_cat1_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat1])] for i in d_cat1]), axis=0)[:500]
deaths_cat2_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat2])] for i in d_cat2]), axis=0)[:500]
deaths_cat3_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat3])] for i in d_cat3]), axis=0)[:500]
deaths_cat4_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat4])] for i in d_cat4]), axis=0)[:500]
deaths_cat5_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat5])] for i in d_cat5]), axis=0)[:500]
deaths_cat6_sd = np.std(np.array([i[:np.min([len(i) for i in d_cat6])] for i in d_cat6]), axis=0)[:500]

fig, axs = plt.subplots(6, 1, figsize=(4, 15), sharex=True)  
axs[0].plot(creation_cat1_mean, color='tab:blue', linewidth = 2)
# axs[0].plot(creation_cat1_mean+creation_cat1_sd, color='tab:blue', linestyle = '--')
# axs[0].plot(creation_cat1_mean-creation_cat1_sd, color='tab:blue', linestyle = '--')
axs[0].plot(deaths_cat1_mean, color = 'tab:orange', linewidth = 2)
# axs[0].plot(deaths_cat1_mean+deaths_cat1_sd, color='tab:orange', linestyle = '--')
# axs[0].plot(deaths_cat1_mean-deaths_cat1_sd, color='tab:orange', linestyle = '--')
axs[0].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat1]))), creation_cat1_mean[:500]+creation_cat1_sd, creation_cat1_mean-creation_cat1_sd, alpha = 0.2)
axs[0].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat1]))), deaths_cat1_mean[:500]+deaths_cat1_sd, deaths_cat1_mean-deaths_cat1_sd, alpha = 0.2)


axs[1].plot(creation_cat2_mean, color='tab:blue', linewidth = 2)
# axs[1].plot(creation_cat2_mean+creation_cat2_sd, color='tab:blue', linestyle = '--')
# axs[1].plot(creation_cat2_mean-creation_cat2_sd, color='tab:blue', linestyle = '--')
axs[1].plot(deaths_cat2_mean, color = 'tab:orange', linewidth = 2)
# axs[1].plot(deaths_cat2_mean+deaths_cat2_sd, color='tab:orange', linestyle = '--')
# axs[1].plot(deaths_cat2_mean-deaths_cat2_sd, color='tab:orange', linestyle = '--')
axs[1].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat2]))), creation_cat2_mean+creation_cat2_sd, creation_cat2_mean-creation_cat2_sd, alpha = 0.2)
axs[1].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat2]))), deaths_cat2_mean+deaths_cat2_sd, deaths_cat2_mean-deaths_cat2_sd, alpha = 0.2)


axs[2].plot(creation_cat3_mean, color='tab:blue', linewidth = 2)
# axs[2].plot(creation_cat3_mean+creation_cat3_sd, color='tab:blue', linestyle = '--')
# axs[2].plot(creation_cat3_mean-creation_cat3_sd, color='tab:blue', linestyle = '--')
axs[2].plot(deaths_cat3_mean, color = 'tab:orange', linewidth = 2)
# axs[2].plot(deaths_cat3_mean+deaths_cat3_sd, color='tab:orange', linestyle = '--')
# axs[2].plot(deaths_cat3_mean-deaths_cat3_sd, color='tab:orange', linestyle = '--')
axs[2].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat3]))), creation_cat3_mean+creation_cat3_sd, creation_cat3_mean-creation_cat3_sd, alpha = 0.2)
axs[2].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat3]))), deaths_cat3_mean+deaths_cat3_sd, deaths_cat3_mean-deaths_cat3_sd, alpha = 0.2)


axs[3].plot(creation_cat4_mean, color='tab:blue', linewidth = 2)
# axs[3].plot(creation_cat4_mean+creation_cat4_sd, color='tab:blue', linestyle = '--')
# axs[3].plot(creation_cat4_mean-creation_cat4_sd, color='tab:blue', linestyle = '--')
axs[3].plot(deaths_cat4_mean, color = 'tab:orange', linewidth = 2)
# axs[3].plot(deaths_cat4_mean+deaths_cat4_sd, color='tab:orange', linestyle = '--')
# axs[3].plot(deaths_cat4_mean-deaths_cat4_sd, color='tab:orange', linestyle = '--')
axs[3].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat4]))), creation_cat4_mean+creation_cat4_sd, creation_cat4_mean-creation_cat4_sd, alpha = 0.2)
axs[3].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat4]))), deaths_cat4_mean+deaths_cat4_sd, deaths_cat4_mean-deaths_cat4_sd, alpha = 0.2)


axs[4].plot(creation_cat5_mean, color='tab:blue', linewidth = 2)
# axs[4].plot(creation_cat5_mean+creation_cat5_sd, color='tab:blue', linestyle = '--')
# axs[4].plot(creation_cat5_mean-creation_cat5_sd, color='tab:blue', linestyle = '--')
axs[4].plot(deaths_cat5_mean, color = 'tab:orange', linewidth = 2)
# axs[4].plot(deaths_cat5_mean+deaths_cat5_sd, color='tab:orange', linestyle = '--')
# axs[4].plot(deaths_cat5_mean-deaths_cat5_sd, color='tab:orange', linestyle = '--')
axs[4].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat5]))), creation_cat5_mean+creation_cat5_sd, creation_cat5_mean-creation_cat5_sd, alpha = 0.2)
axs[4].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat5]))), deaths_cat5_mean+deaths_cat5_sd, deaths_cat5_mean-deaths_cat5_sd, alpha = 0.2)


axs[5].plot(creation_cat6_mean, color='tab:blue', linewidth = 2)
# axs[5].plot(creation_cat6_mean+creation_cat6_sd, color='tab:blue', linestyle = '--')
# axs[5].plot(creation_cat6_mean-creation_cat6_sd, color='tab:blue', linestyle = '--')
axs[5].plot(deaths_cat6_mean, color = 'tab:orange', linewidth = 2)
# axs[5].plot(deaths_cat6_mean+deaths_cat6_sd, color='tab:orange', linestyle = '--')
# axs[5].plot(deaths_cat6_mean-deaths_cat6_sd, color='tab:orange', linestyle = '--')
axs[5].fill_between(np.arange(min(500, np.min([len(i) for i in c_cat6]))), creation_cat6_mean+creation_cat6_sd, creation_cat6_mean-creation_cat6_sd, alpha = 0.2)
axs[5].fill_between(np.arange(min(500, np.min([len(i) for i in d_cat6]))), deaths_cat6_mean+deaths_cat6_sd, deaths_cat6_mean-deaths_cat6_sd, alpha = 0.2)
for i in range(6):
    axs[i].set_yticks([0, 5, 10])
    axs[i].set_yticklabels([0, 5, 10], fontsize = 14)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
times = [0, 0.5, 1, 1.5, 2]
frames = (3600/15)*(np.array(times))
axs[5].set_xticks(frames)
axs[5].set_xticklabels(times, fontsize = 14)

plt.tight_layout()
plt.show()


### CLOSEST NEIGHBORS

def closest_neighbour_distances(points):
    """Find the distance to the closest neighbour for each point using a KD-Tree."""
    tree = KDTree(points)
    # query the closest point for each point in the dataset; k=2 to get the first neighbor (ignoring the point itself)
    distances, _ = tree.query(points, k=2)
    # distances[:, 1] because the closest point is the point itself at distances[:, 0]
    return distances[:, 1]

mean_closest_distances = []
sd_closest_distances = []
for j, date in enumerate(all_data):
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    
    earliest_start_frame = singularities['frame'].min()
    singularity_start_frames = []
    for i in singularities['particle'].unique():
        singularity_start_frames.append(list(singularities[singularities['particle']==i]['frame'])[0])
    points = np.array([(list(singularities[singularities['frame']==int(np.percentile(singularity_start_frames, 95))+20]['x'])[i], list(singularities[singularities['frame']==int(np.percentile(singularity_start_frames, 95))+20]['y'])[i]) for i in range(len(singularities[singularities['frame']==int(np.percentile(singularity_start_frames, 95))+20]))])
    closest_distances = closest_neighbour_distances(points)
    mean_closest_distances.append(closest_distances.mean()*8*5.5/1000)
    sd_closest_distances.append(sem(closest_distances)*8*5.5/1000)

plt.errorbar(densities, mean_closest_distances, yerr = sd_closest_distances, fmt = 'o', ecolor='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Mean distance between singularities and \n their closest neighbour (mm)")
z = np.polyfit(densities, mean_closest_distances, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, mean_closest_distances)
print(r_value**2, p_value)


# FIG 2I
fig=plt.figure(figsize  =(7,7))
plt.scatter(densities, peaks, label = "Max", c='tab:blue')
plt.scatter(densities, singularities_to_be_pruned, alpha = 1, label = "Stable", c='tab:orange')
plt.scatter(densities, aggregates, alpha = 1, label = "Final", c="tab:red")
plt.vlines(densities, ymin=aggregates, ymax = peaks, color='black', alpha = 0.2, zorder =0)
plt.xticks([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], fontsize = 13)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70], fontsize = 13)
# plt.ylabel("Number of singularities")
# plt.xlabel("Initial cell density ($cells/mm^2$)")
# plt.legend(loc = 'upper left')
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()



fig=plt.figure(figsize = (10,3))
plt.hist((np.array(earliest_start_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), edgecolor = 'tab:olive', density = True, facecolor='None', alpha = 0.3)
plt.hist((np.array(earliest_start_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), alpha = 0.1, density = True, facecolor='tab:olive')
mean = np.nanmean((np.array(earliest_start_frames)-np.array(synchronisation_frame))/4)
sd = np.nanstd((np.array(earliest_start_frames)-np.array(synchronisation_frame))/4)
plt.plot(np.arange(-40, 80, 0.1), norm.pdf(np.arange(-40, 80, 0.1), mean, sd), color = 'tab:olive')

plt.hist((np.array(peak_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), edgecolor = 'tab:cyan', density=True, facecolor = 'None', alpha = 0.3)
plt.hist((np.array(peak_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), density=True, facecolor = 'tab:cyan', alpha = 0.1)
mean = np.nanmean((np.array(peak_frames)-np.array(synchronisation_frame))/4)
sd = np.nanstd((np.array(peak_frames)-np.array(synchronisation_frame))/4)
plt.plot(np.arange(-40, 80, 0.1), norm.pdf(np.arange(-40, 80, 0.1), mean, sd), color = 'tab:cyan')

plt.hist((np.array(stabilising_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), edgecolor = 'tab:purple', density=True, facecolor = "None", alpha = 0.3)
plt.hist((np.array(stabilising_frames)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), density=True, facecolor = "tab:purple", alpha = 0.1)
mean = np.nanmean((np.array(stabilising_frames)-np.array(synchronisation_frame))/4)
sd = np.nanstd((np.array(stabilising_frames)-np.array(synchronisation_frame))/4)
plt.plot(np.arange(-40, 80, 0.1), norm.pdf(np.arange(-40, 80, 0.1), mean, sd), color = 'tab:purple')

plt.hist((np.array(creation_peaks)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), edgecolor = 'tab:green', density=True, facecolor = "None", alpha = 0.3)
plt.hist((np.array(creation_peaks)-np.array(synchronisation_frame))/4, bins=np.arange(-45, 95, 5), alpha = 0.1, density=True, facecolor = "tab:green")
mean = np.nanmean((np.array(creation_peaks)-np.array(synchronisation_frame))/4)
sd = np.nanstd((np.array(creation_peaks)-np.array(synchronisation_frame))/4)
plt.plot(np.arange(-40, 80, 0.1), norm.pdf(np.arange(-40, 80, 0.1), mean, sd), color = 'tab:green')
axs=fig.gca()
for axis in ['top','bottom','left','right']:
    axs.spines[axis].set_linewidth(2)
plt.xticks([-40, -20, 0, 20, 40, 60, 80], fontsize = 13)
plt.yticks([0, 0.05, 0.1], fontsize = 13)
plt.show()




# # Get cyclic colormap
# azimuths = np.arange(0, 361, 1)
# zeniths = np.arange(40, 70, 1)
# values = azimuths * np.ones((30, 361))

# fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# # Converting azimuths to radians for plotting
# radian_azimuths = azimuths * np.pi / 180.0
# ax.pcolormesh(radian_azimuths, zeniths, values, cmap=plt.cm.twilight)

# # Setting the 0 radians to the left (west)
# ax.set_theta_zero_location('N')

# # Setting theta to increase clockwise
# ax.set_theta_direction(-1)

# # Define custom ticks to match the desired description
# custom_ticks = np.pi * np.array([])
# # custom_labels = [r'$\pi$', r'$-\frac{3\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
# custom_labels = []
# ax.set_xticks(custom_ticks)
# ax.set_xticklabels(custom_labels, fontsize  = 10)

# # Hide y-axis labels as before
# ax.set_yticks([])
# plt.text(0.5, 0.2, '0', transform=plt.gcf().transFigure, fontsize = 30, color = 'white')
# plt.text(0.46, 0.75, r'±$\pi$', transform=plt.gcf().transFigure, fontsize = 30, color = 'black')
# plt.show()


