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

frame_intervals = 15

control_data = {
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

caffeine_data = {
    "C-24-06-01": [7781, 21, 0, 2.5, 0.25],
    "C-24-06-02": [7092, 29, 0, 2.5, 0.25],
    "C-24-06-04": [7673, 35, 0, 2.5, 0.25], 
    "C-24-06-05": [8846, 22, 0, 2, 0.25],
    "C-24-05-24": [6644, 8, 0, 2+23/60, 0.5],
    "C-24-05-29": [5391, 8, 0, 2, 0.5],
    "C-24-05-30-0.5": [6601, 14, 0, 2+1/12, 0.5], 
    "C-24-05-31-0.5": [6875, 11, 2, 2.5, 0.5],
    # "C-24-05-22": [7269, 5, 1, 2.5, 1],
    "C-24-05-28": [6745, 11, 0, 2, 1],
    "C-24-05-30-1": [7038, 10, 0, 2, 1],
    "C-24-05-31-1": [6690, 13, 0, 2, 1]
    }


all_data=dict(control_data)
all_data.update(caffeine_data)

densities = [i[0] for i in list(all_data.values())]
aggregates = [i[1] for i in list(all_data.values())]
late_prune = [i[2] for i in list(all_data.values())]
start_times = [i[3] for i in list(all_data.values())]

#### SINGULARITY TIME PLOTS
singularities = []
mean_singularities = []
peaks = []
final_number = []
for i in all_data:
    s = get_n_singularities(i)

    mean_s = smooth_singularity_time_series(s)

    singularities.append(s)
    mean_singularities.append(mean_s)
    peaks.append(np.max(s))



    

colormap = plt.cm.viridis  
num_plots = len(all_data)

# Generating colors from the colormap
color_indices = np.linspace(0, 1, num_plots)
colors = [colormap(i) for i in color_indices]

caffeine_color = {
    0.25: 'tab:red',
    0.5: 'tab:green',
    1: 'tab:blue'
    }


plt.figure(figsize = (10,5))
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


fig, ax = plt.subplots(figsize=(10, 5))  

for i, data in enumerate(mean_singularities):
    if list(all_data)[i] == "S-24-04-11" or list(all_data)[i] == "S-24-04-25-PM":
        continue
    ax.plot(data[:350], c=colors[i], label=densities[i])

ax.set_xlabel("Time since first singularity formed (h)", fontsize=12)
times = [0, 0.5, 1, 1.5]
frames = (3600/15)*(np.array(times))
ax.set_xticks(frames)
ax.set_xticklabels(times)
ax.set_ylabel("Number of singularities", fontsize=12)

sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])
labels = densities
cbar.set_ticklabels([2000, 9000])  
cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)
plt.show()


plt.scatter(densities[:-len(caffeine_data)], peaks[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], peaks[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Maximum number of singularities");
z = np.polyfit(densities[:-len(caffeine_data)], peaks[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()



plt.scatter(densities[:-len(caffeine_data)], aggregates[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], aggregates[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()

# CREATION

# Plot histogram of singularity creation times
all_start_times = []
relative_start_times = []
super_sing_starts = []
singularities_to_be_pruned = []
stabilising_times = []
singularity_creation_rate = []
for j, date in enumerate(all_data):
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    
    earliest_start_time = singularities['frame'].min()
    singularity_start_times = []
    for i in singularities['particle'].unique():
        singularity_start_times.append(list(singularities[singularities['particle']==i]['frame'])[0])
        if date!="S-24-04-11" and date!= "S-24-04-25-PM":
            all_start_times.append(list(singularities[singularities['particle']==i]['frame'])[0])
        relative_start_times.append(list(singularities[singularities['particle']==i]['frame'])[0]-earliest_start_time)
        if list(singularities[singularities['particle']==i]['frame'])[-1] ==max(singularities['frame']) and date!="S-24-04-11" and date!= "S-24-04-25-PM":
            super_sing_starts.append(list(singularities[singularities['particle']==i]['frame'])[0])
    singularities_to_be_pruned.append(len(singularities[singularities['frame']==int(np.percentile(singularity_start_times, 95))+20]))
    stabilising_times.append(int(np.percentile(singularity_start_times, 95))+20)
    singularity_creation_rate.append(len(singularities['particle'].unique())/((np.max(singularity_start_times)-np.min(singularity_start_times))/4))
relative_start_times = np.array(relative_start_times)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/15)*(np.array(times)-3)
plt.hist(all_start_times, density = True, bins = 30, label = "All singularities");
plt.hist(super_sing_starts, density = True, bins = 30, color='red', alpha =0.5, label = 'Supersingularities')
plt.xticks(frames, times);
plt.xlabel("Birth time after starvation (h)", fontsize = 12);
plt.ylabel("Probability density", fontsize = 12)
plt.legend()
plt.show()

plt.scatter(densities, singularity_creation_rate, c='black')
plt.ylim([0, 10])
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Number of singularities created per minute")
z = np.polyfit(densities, singularity_creation_rate, 1)
p = np.poly1d(z)
plt.plot(densities,p(densities),"r--", alpha =0.4)
plt.show()
slope, intercept, r_value, p_value, std_err = linregress(densities, singularity_creation_rate)
print(r_value**2, p_value)



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

plt.scatter(densities[:-len(caffeine_data)], end_periods_sd[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], end_periods_sd[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Standard deviation of periods of supersingularities")
z = np.polyfit(densities[:-len(caffeine_data)], end_periods_sd[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()

periods_across_time_mean = []
periods_across_time_sd = []
for i, date in enumerate(all_data):
    singularities = pd.read_pickle(f"Data/{date}/Analysed_data/singularities.pkl")
    period_sd = []
    period_mean = []
    for frame in np.arange(stabilising_times[i], singularities['frame'].max()):

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


fig, ax = plt.subplots(figsize=(10, 5))  
for i in range(len(control_data)):
    ax.plot(periods_across_time_mean[i], c=colors[i])
ax.set_xlabel("Frames since stabilisation")
ax.set_ylabel("Mean angular period of all singularities (mins)")
sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])
cbar.set_ticklabels([2000, 9000])  

cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)


for j in range(len(caffeine_data)):
    ax.plot(periods_across_time_mean[len(control_data)+j], c=caffeine_color[list(caffeine_data.values())[j][4]], linestyle ='--', linewidth = 5)

plt.show(cbar)



times = np.array([3, 4, 5, 6, 7])
fig, ax = plt.subplots(figsize=(10, 5))  
for i in range(len(control_data)):
    start_frame = list(control_data.values())[i][3]*60*60/15
    ax.plot(np.arange(start_frame+stabilising_times[i]+len(periods_across_time_mean[i])), [np.nan]*int(start_frame+stabilising_times[i])+periods_across_time_mean[i], c=colors[i])
ax.set_xlabel("Time since starvation (hr)")
ax.set_xticks(times*4*60)
ax.set_xticklabels(times)
ax.set_ylabel("Mean angular period of all singularities (mins)")
sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])
cbar.set_ticklabels([2000, 9000])  

cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)

for j in range(len(caffeine_data)):
    start_frame = list(caffeine_data.values())[j][3]*60*60/15
    ax.plot(np.arange(start_frame+stabilising_times[len(control_data)+j]+len(periods_across_time_mean[len(control_data)+j])),[np.nan]*int(start_frame+stabilising_times[len(control_data)+j])+periods_across_time_mean[len(control_data)+j], c=caffeine_color[list(caffeine_data.values())[j][4]], linestyle ='--', linewidth = 5, alpha = 0.5)

plt.show(cbar)


times = np.array([3, 4, 5, 6, 7])
fig, ax = plt.subplots(figsize=(10, 5))  
for i in range(len(control_data)):
    start_frame = list(control_data.values())[i][3]*60*60/15
    ax.plot(np.arange(start_frame+stabilising_times[i]+len(periods_across_time_mean[i])), [np.nan]*int(start_frame+stabilising_times[i])+periods_across_time_mean[i], c=colors[i])
# ax.set_xlabel("Time since starvation (hr)")
ax.set_xticks(times*4*60)
ax.set_xticklabels(times)
ax.set_ylabel("Mean angular period of all singularities (mins)")
sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])
cbar.set_ticklabels([2000, 9000])  

cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)
plt.show(cbar)



cats = np.full((3*60*4+900, len(control_data)), np.nan)
for i in range(len(control_data)):
    start_frame = list(control_data.values())[i][3]*60*60/15
    cats[int(start_frame+stabilising_times[i]):int(start_frame+stabilising_times[i])+len(periods_across_time_mean[i]), i] = periods_across_time_mean[i]
for i in range(len(control_data)):
    if densities[i]>1000 and densities[i]<3000:
        # cats1 = cats[:, :i]
        # mask = ~np.isnan(cats1).any(axis=1)
        # cat1 = np.where(mask, np.mean(cats1, axis=1), np.nan)
        # std1 = np.where(mask, np.std(cats1, axis=1), np.nan)
        cat1 = np.nanmean(cats[:, :i], axis=1)
        std1 = np.nanstd(cats[:, :i], axis=1)
        cat1[:1155]=[np.nan]*1155
        cat1[1550:]=[np.nan]*len(cat1[1550:])
        a = i
    if densities[i]>3000 and densities[i]<4000:
        # cats2 = cats[:, a:i]
        # mask = ~np.isnan(cats2).any(axis=1)
        # cat2 = np.where(mask, np.mean(cats2, axis=1), np.nan)
        # std2 = np.where(mask, np.std(cats2, axis=1), np.nan)
        cat2 = np.nanmean(cats[:, a:i], axis=1)
        std2 = np.nanstd(cats[:, a:i], axis=1)
        cat2[:1040] = [np.nan]*1040
        b=i
    if densities[i]>4000 and densities[i]<5000:
        # cats3 = cats[:, b:i]
        # mask = ~np.isnan(cats3).any(axis=1)
        # cat3 = np.where(mask, np.mean(cats3, axis=1), np.nan)
        # std3 = np.where(mask, np.std(cats3, axis=1), np.nan)
        cat3 = np.nanmean(cats[:, b:i], axis=1)
        std3 = np.nanstd(cats[:, b:i], axis=1)
        cat3[:1020] = [np.nan]*1020
        c=i
    if densities[i]>5000 and densities[i]<6000:
        # cats4 = cats[:, c:i]
        # mask = ~np.isnan(cats4).any(axis=1)
        # cat4 = np.where(mask, np.mean(cats4, axis=1), np.nan)
        # std4 = np.where(mask, np.std(cats4, axis=1), np.nan)
        cat4 = np.nanmean(cats[:, c:i], axis=1)
        std4 = np.nanstd(cats[:, c:i], axis=1)
        d=i
    if densities[i]>6000 and densities[i]<7000:
        # cats5 = cats[:, d:i]
        # mask = ~np.isnan(cats5).any(axis=1)
        # cat5 = np.where(mask, np.mean(cats5, axis=1), np.nan)
        # std5 = np.where(mask, np.std(cats5, axis=1), np.nan)
        cat5 = np.nanmean(cats[:, d:i], axis=1)
        std5 = np.nanstd(cats[:, d:i], axis=1)
        cat5[:940] = [np.nan]*940
        e=i
    if densities[i]>7000:
        # cats6 = cats[:, e:i]
        # mask = ~np.isnan(cats6).any(axis=1)
        # cat6 = np.where(mask, np.mean(cats6, axis=1), np.nan)
        # std6 = np.where(mask, np.std(cats6, axis=1), np.nan)
        cat6 = np.nanmean(cats[:, e:i], axis=1)
        std6 = np.nanstd(cats[:, e:i], axis=1)
        cat6[:900] = [np.nan]*900
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

caff_colors = ["#fcae91",
"#fb6a4a",
"#de2d26",
"#a50f15",
"#bae4b3",
"#74c476",
"#31a354",
"#006d2c",
"#6baed6",
"#3182bd",
"#08519c"]
times = np.array([3, 4, 5, 6, 7])
fig, ax = plt.subplots(figsize=(10, 7))  
for j in range(len(caffeine_data)):
    start_frame = list(caffeine_data.values())[j][3]*60*60/15
    ax.plot(np.arange(start_frame+stabilising_times[len(control_data)+j]+len(periods_across_time_mean[len(control_data)+j])),[np.nan]*int(start_frame+stabilising_times[len(control_data)+j])+periods_across_time_mean[len(control_data)+j], linewidth = 3, alpha = 0.7,c = caff_colors[j])
ax.plot(cat5, c='black', linewidth = 3, label = "6000-7000")
# ax.plot(cat5+std5, c='black', linewidth = 1)
# ax.plot(cat5-std5, c='black', linewidth = 1)
ax.fill_between(np.arange(len(cat5)), cat5+std5, cat5-std5, alpha = 0.2, color = 'black')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(axis='y', labelsize = 13)
ax.tick_params(axis='x', labelsize = 13)
plt.xticks(times*4*60, times)
plt.xlim([600, 1700])
plt.show()


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

plt.errorbar(densities[:-len(caffeine_data)], mean_all_speed[:-len(caffeine_data)], yerr = sd_all_speed[:-len(caffeine_data)], fmt = 'o', color = 'black', ecolor='black', label = '0')
for i in range(len(caffeine_data)): 
    plt.errorbar(densities[len(control_data)+i], mean_all_speed[len(control_data)+i], yerr = sd_all_speed[len(control_data)+i], fmt = 'o', c=caffeine_color[list(caffeine_data.values())[i][4]], label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)", size = 12)
plt.ylabel("Mean speed of all singularities (mm/min)")
z = np.polyfit(densities[:-len(caffeine_data)], mean_all_speed[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
plt.show()

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


plt.scatter(densities[:-len(caffeine_data)], all_n_pairs[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], all_n_pairs[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of pair annihilation events")
plt.yticks([0, 5, 10, 15, 20])
z = np.polyfit(densities[:-len(caffeine_data)], all_n_pairs[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
plt.show()


plt.scatter(densities[:-len(caffeine_data)], all_premature[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], all_premature[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of premature deaths")
z = np.polyfit(densities[:-len(caffeine_data)], all_premature[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()


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
plt.scatter(densities[:-len(caffeine_data)], late_prune[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], late_prune[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of motility-induced deaths")
z = np.polyfit(densities[:-len(caffeine_data)], late_prune[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()



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

for i in range(len(densities)-len(caffeine_data)):
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
        all_start_times = list(singularities[singularities['particle']==i]['frame'])[0]
        creation.append(all_start_times)
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
prune_diffs = []
for i, date in enumerate(all_data):
    
    label_distribution=np.load(f"Data/{date}/Analysed_data/correlation_areas.npy")
    synchronisation_frame.append(np.argmax(label_distribution==label_distribution.max()))
   

    

    creation_rates = all_creation_plots[i]
    death_rates = all_death_plots[i]
    creation_diffs.append(np.argmax(creation_rates)-synchronisation_frame[i])
    prune_diffs.append(np.argmax(death_rates)-synchronisation_frame[i])

    # print(densities[i], synchronisation_frame[i])
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Creating two subplots with shared x-axis
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
    
plt.scatter(densities, np.array(creation_diffs)/4, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Difference between time of peak creation rate \nand development (mins)")
plt.show()


singularity_creation_range = [(stabilising_times[i]-synchronisation_frame[i])/4 for i in range(len(all_data))]
plt.scatter(densities[:-len(caffeine_data)], singularity_creation_range[:-len(caffeine_data)], c='black', s=10, label = '0')
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], singularity_creation_range[len(control_data)+i], c=caffeine_color[list(caffeine_data.values())[i][4]], s=10, label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Difference between stabilisation \n and development time (mins)")
z = np.polyfit(densities[:-len(caffeine_data)], singularity_creation_range[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
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
    
    earliest_start_time = singularities['frame'].min()
    singularity_start_times = []
    for i in singularities['particle'].unique():
        singularity_start_times.append(list(singularities[singularities['particle']==i]['frame'])[0])
    points = np.array([(list(singularities[singularities['frame']==int(np.percentile(singularity_start_times, 95))+20]['x'])[i], list(singularities[singularities['frame']==int(np.percentile(singularity_start_times, 95))+20]['y'])[i]) for i in range(len(singularities[singularities['frame']==int(np.percentile(singularity_start_times, 95))+20]))])
    closest_distances = closest_neighbour_distances(points)
    mean_closest_distances.append(closest_distances.mean()*8*5.5/1000)
    sd_closest_distances.append(sem(closest_distances)*8*5.5/1000)

plt.errorbar(densities[:-len(caffeine_data)], mean_closest_distances[:-len(caffeine_data)], yerr = sd_closest_distances[:-len(caffeine_data)], fmt = 'o', color = 'black', ecolor='black', label = '0')
for i in range(len(caffeine_data)): 
    plt.errorbar(densities[len(control_data)+i], mean_closest_distances[len(control_data)+i], yerr = sd_closest_distances[len(control_data)+i], fmt = 'o', c=caffeine_color[list(caffeine_data.values())[i][4]], label = list(caffeine_data.values())[i][4])
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Mean distance between singularities and \n their closest neighbour (mm)")
z = np.polyfit(densities[:-len(caffeine_data)], mean_closest_distances[:-len(caffeine_data)], 1)
p = np.poly1d(z)
plt.plot(densities[:-len(caffeine_data)],p(densities[:-len(caffeine_data)]),"r--", alpha =0.4)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()

fig=plt.figure(figsize = (7, 9))
kuramotos = []
for j, date in enumerate(all_data):
    kuramoto = np.load(f"Data/{date}/Analysed_data/kuramoto.npy")
    if kuramoto[:500].argmin()==499:
        kuramotos.append(np.nan)
    else:
        kuramotos.append(kuramoto[:500].min())
plt.scatter(densities[:-len(caffeine_data)], kuramotos[:-len(caffeine_data)], c='black', label = '0', s=60)
for i in range(len(caffeine_data)): 
    plt.scatter(densities[len(control_data)+i], kuramotos[len(control_data)+i], c=caff_colors[i], label = list(caffeine_data.values())[i][4], s=400, marker = '*')
plt.xticks([2000, 4000, 6000, 8000, 10000], fontsize = 14)
plt.yticks(fontsize = 14)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
idx = np.isfinite(densities[:-len(caffeine_data)]) & np.isfinite(kuramotos[:-len(caffeine_data)])
# plt.xlabel("Initial cell density ($cells/mm^2$)")
# plt.ylabel("Kuramoto order parameter")
# plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()

fig=plt.figure(figsize = (7, 9))
plt.scatter(kuramotos[:-len(caffeine_data)], singularities_to_be_pruned[:-len(caffeine_data)], c='black', label = '0', s=60)
for i in range(len(caffeine_data)): 
    plt.scatter(kuramotos[len(control_data)+i], singularities_to_be_pruned[len(control_data)+i], c=caff_colors[i], label = list(caffeine_data.values())[i][4], s=400, marker = '*')
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
idx = np.isfinite(kuramotos[:-len(caffeine_data)]) & np.isfinite(singularities_to_be_pruned[:-len(caffeine_data)])
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.xlabel("Kuramoto order parameter")
# plt.ylabel("Number of singularities at stabilisation frame")
# plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()


plt.scatter(densities, singularities_to_be_pruned, label = "After premature deaths", c='tab:blue')
plt.scatter(densities, final_number, alpha = 1, label = "After premature and pair deaths", c='tab:orange')
plt.scatter(densities, aggregates, alpha = 1, label = "Final number of aggregates", c="tab:red")

plt.ylabel("Number of singularities")
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylim([0, 55])
plt.legend(loc = 'upper left')
plt.show()



wave_period = []
for data in tqdm(all_data):
        periods = np.load(f"Data/{data}/Analysed_data/periods.npy")
        wave_period.append(np.nanmean(periods))

fig=plt.figure(figsize=(7, 9))
plt.scatter(densities[:len(control_data)], wave_period[:len(control_data)], c='black', label = '0', s=60)
for i in range(len(caffeine_data)): 

    plt.scatter(densities[len(control_data)+i], wave_period[len(control_data)+i], c=caff_colors[i], label = list(caffeine_data.values())[i][4], s=400, marker = '*')
# plt.xlabel("Initial cell density ($cells/mm^2$)");
# plt.ylabel("Wave period during synchronisation (mins)")
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.xticks([2000, 4000, 6000, 8000, 10000], fontsize = 14)
plt.yticks(fontsize = 14)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.legend(unique_labels.values(), unique_labels.keys())
plt.show()

fig=plt.figure(figsize = (7, 9))
plt.scatter(wave_period[:len(control_data)], singularities_to_be_pruned[:len(control_data)], c='black', label = '0', s = 60)
for i in range(len(caffeine_data)): 

    plt.scatter(wave_period[len(control_data)+i], singularities_to_be_pruned[len(control_data)+i], c=caff_colors[i], label = list(caffeine_data.values())[i][4], s=400, marker = '*')
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax=fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
# plt.legend(unique_labels.values(), unique_labels.keys())
# plt.ylabel("Number of singularities at stabilisation");
# plt.xlabel("Wave period during synchronisation (mins)")
plt.show()
