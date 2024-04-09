import numpy as np
from src.animate import *
from src.analysis_utils import *
from src.helpers import *
import pandas as pd
from scipy.signal import savgol_filter

import pingouin as pg


frame_intervals = 15
total_frames = 900

date_densities = {
    "S-24-01-12-PM": 4091,
    "S-24-04-03": 4460,
    "S-24-02-07-AM": 4493,
    "S-24-02-02": 4787,
    "S-24-01-24": 5047,
    "S-24-01-23": 5080,
    "S-24-01-30-PM": 5381,
    "S-24-01-17-PM": 5388,
    "S-24-02-06-PM": 5394,
    "S-24-02-07-PM": 5491,
    "S-24-01-25": 5667,
    "S-24-01-30-AM": 5772,
    "S-24-02-06-AM": 5970,
    "S-24-02-01": 5994,
    "S-24-01-11": 6031,
    "S-24-01-12-AM": 6216,
    "S-24-01-16": 6302,
    "S-24-01-17-AM": 6358,
    "S-24-01-31": 6467,
    "S-24-01-10": 6586,
    "S-24-01-26": 7376    
}

date_aggregates = {
    "S-24-01-12-PM": 7,
    "S-24-04-03": 7, 
    "S-24-02-07-AM": 7,
    "S-24-02-02": 12 ,
    "S-24-01-24": 4,
    "S-24-01-23": 7,
    "S-24-01-30-PM": 14,
    "S-24-01-17-PM": 4,
    "S-24-02-06-PM": 15,
    "S-24-02-07-PM": 6,
    "S-24-01-25": 8,
    "S-24-01-30-AM": 6,
    "S-24-02-06-AM": 8,
    "S-24-02-01": 9,
    "S-24-01-11": 13,
    "S-24-01-12-AM": 13,
    "S-24-01-16": 13,
    "S-24-01-17-AM": 8,
    "S-24-01-31": 13,
    "S-24-01-10": 11,
    "S-24-01-26": 13    
}

date_late_prune = {
    "S-24-01-12-PM": 4,
    "S-24-04-03": 3,
    "S-24-02-07-AM": 3,
    "S-24-02-02": 3,
    "S-24-01-24": 4,
    "S-24-01-23": 4,
    "S-24-01-30-PM": 3,
    "S-24-01-17-PM": 5,
    "S-24-02-06-PM": 0,
    "S-24-02-07-PM": 0,
    "S-24-01-25": 2,
    "S-24-01-30-AM": 0,
    "S-24-02-06-AM": 3,
    "S-24-02-01": 5,
    "S-24-01-11": 2,
    "S-24-01-12-AM": 1,
    "S-24-01-16": 0,
    "S-24-01-17-AM": 1,
    "S-24-01-31": 2,
    "S-24-01-10": 0,
    "S-24-01-26": 1    
}


#### SINGULARITY TIME PLOTS
singularities = []
mean_singularities = []
peaks = []
final_number = []
for i in date_densities:
    s = get_n_singularities(i)

    # S-24-01-30 started 10 mins late
    if i == "S-24-01-30-PM":
        s=np.append([0]*40,s)
    # S-24-02-06-AM started 20 mins late
    elif i == "S-24-02-06-AM":
        s=np.append([0]*80,s)
    mean_s = smooth_singularity_time_series(s)

    singularities.append(s)
    mean_singularities.append(mean_s)
    peaks.append(np.max(s))


densities = date_densities.values()
    

colormap = plt.cm.viridis  # A colormap with a good range from light to dark
num_plots = len(date_densities)  # Number of scatter plots

# Generating colors from the colormap
color_indices = np.linspace(0, 1, num_plots)
colors = [colormap(i) for i in color_indices]


plt.figure(figsize = (10,5))
for i, data in enumerate(singularities):
    final_number.append(data[np.array(data)!=0][-1])
    if list(date_densities)[i] == "S-24-01-30-PM":
        plt.scatter(np.arange(450), data[:450], s=1, alpha=0.6, c=[colors[i]])
    else:
        plt.scatter(np.arange(500), data[:500], s=1, alpha=0.6, c=[colors[i]])
        

for i, data in enumerate(mean_singularities):
    if list(date_densities)[i] == "S-24-01-30-PM":
        plt.plot(data[:450], c = colors[i], label = date_densities[list(date_densities)[i]])
    else:
        plt.plot(data[:500], c = colors[i], label = date_densities[list(date_densities)[i]])

plt.xlabel("Time after starvation (h)", fontsize = 12)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/15)*(np.array(times)-3)
plt.xticks(frames, times)
plt.ylabel("Number of singularities", fontsize = 12)
plt.legend(loc='upper right', fontsize = 7)
plt.show()


mean_singularities = []
ratios = []
for i, s in enumerate(singularities):

    mean_s = smooth_singularity_time_series(s, since_birth=True)
    mean_singularities.append(mean_s)
    ratios.append(date_aggregates[list(date_densities)[i]]/np.nanmax(mean_s))


fig, ax = plt.subplots(figsize=(10, 5))  

for i, data in enumerate(mean_singularities):
    ax.plot(data[:350], c=colors[i], label=date_densities[list(date_densities)[i]])

ax.set_xlabel("Time since first singularity formed (h)", fontsize=12)
times = [0, 0.5, 1, 1.5]
frames = (3600/15)*(np.array(times))
ax.set_xticks(frames)
ax.set_xticklabels(times)
ax.set_ylabel("Number of singularities", fontsize=12)


sm = ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', ticks=[0, 1])


labels = list(date_densities.values())
cbar.set_ticklabels([4000, 7500])  

cbar.set_label('Initial cell density ($cells/mm^2$)', fontsize=10)

plt.show()


plt.scatter(densities, final_number, s=10, label = "Final number detected by algorithm")
plt.scatter(densities, date_aggregates.values(), s=10, label = 'Final number of aggregates')
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel("Number of singularities");
plt.legend()
plt.show()


plt.scatter(densities, ratios, c='black', s=10)
plt.xlabel("Initial cell density ($cells/mm^2$)");
plt.ylabel(r'$\frac{\text{Final number}}{\text{Maximum number}}$');
plt.show()

#### SINGULARITY PERIODS

aggregates = []
deaths = []
mean_aggregate = []
mean_dead = []
mean_total = []
for date in date_densities:
    positive = pd.read_pickle(f"Data/{date}/Analysed_data/positive_df.pkl")
    negative = pd.read_pickle(f"Data/{date}/Analysed_data/negative_df.pkl")

    aggregate = []
    death = []
    total = []
    for i in positive['particle'].unique():
        if len(positive[positive['particle']==i])>10:
            total.append(np.nanmean(positive[positive['particle']==i]['period'][:30]))
        if len(positive[positive['particle']==i])>200:
            aggregates.append(np.nanmean(positive[positive['particle']==i]['period'][:30]))
            aggregate.append(np.nanmean(positive[positive['particle']==i]['period'][:30]))
        elif len(positive[positive['particle']==i])<50 and len(positive[positive['particle']==i])>10:
            deaths.append(np.nanmean(positive[positive['particle']==i]['period'][:30]))
            death.append(np.nanmean(positive[positive['particle']==i]['period'][:30]))
    for i in negative['particle'].unique():
        if len(negative[negative['particle']==i])>10:
            total.append(np.nanmean(negative[negative['particle']==i]['period'][:30]))
        if len(negative[negative['particle']==i])>200:
            aggregates.append(np.nanmean(negative[negative['particle']==i]['period'][:30]))
            aggregate.append(np.nanmean(negative[negative['particle']==i]['period'][:30]))
        elif len(negative[negative['particle']==i])<50 and len(negative[negative['particle']==i])>10:
            deaths.append(np.nanmean(negative[negative['particle']==i]['period'][:30]))
            death.append(np.nanmean(negative[negative['particle']==i]['period'][:30]))
            
    mean_aggregate.append(np.nanmean(aggregate))
    mean_dead.append(np.nanmean(death))
    mean_total.append(np.nanmean(total))
plt.hist(aggregates, density= True, alpha = 0.9, label = "Aggregate", bins = 30)
plt.hist(deaths, density = True, alpha = 0.6, label = "Pruned", bins = 30)
plt.xlabel("Period of singularities at birth (mins)")
plt.axvline(np.nanmean(aggregates), c='black', linestyle = 'solid', label = "Mean aggregate period")
plt.axvline(np.nanmean(deaths), c='black', linestyle = 'dotted', label = 'Mean pruned period')
plt.ylabel("Density")

plt.legend()
plt.show()


print(np.nanmean(aggregates))
print(np.nanmean(deaths))

plt.figure(figsize = (7,5))
plt.scatter(date_densities.values(), mean_aggregate, label = "Aggregate")
plt.scatter(date_densities.values(), mean_dead, label = "Pruned")
# plt.scatter(date_densities.values(), mean_total, label = "Total")
plt.legend()
plt.xlabel("Initial cell density ($cells/mm^2$)", size = 12)
plt.ylabel("Mean period of singularities at birth \n(mins)", fontsize = 12)

z = np.polyfit(list(date_densities.values()), mean_aggregate, 1)
p = np.poly1d(z)
# plt.plot(list(date_densities.values()),p(list(date_densities.values())),"r--", alpha =0.4)
plt.show()

#### DEATHS
all_n_pairs = []
distances = []
density_distances = []
all_premature = []
total_singularities = []
for date in date_densities:
    positive = pd.read_pickle(f"Data/{date}/Analysed_data/positive_df.pkl")
    negative = pd.read_pickle(f"Data/{date}/Analysed_data/negative_df.pkl")
    pairs = []
    density_distance = []
    premature = []

    n_positive = len(positive['particle'].unique())
    n_negative = len(negative['particle'].unique())
    total_singularities.append(n_positive+n_negative)
    for particle_id in positive['particle'].unique():

        # If singularity died because it made it to the end then ignore
        if list(positive[positive['particle']==particle_id]['frame'])[-1] == list(positive['frame'])[-1]:
            continue

        # Only look at singularities that stabilized for over 20 frames for pair annihilations
        elif len(positive[positive['particle']==particle_id]) > 20:
            pairs.append([list(positive[positive['particle']==particle_id]['frame'])[-1], list(positive[positive['particle']==particle_id]['x'])[-1], list(positive[positive['particle']==particle_id]['y'])[-1]])
        
        # Premature deaths
        else: 
            premature.append(particle_id)

    for particle_id in negative['particle'].unique():
        if list(negative[negative['particle']==particle_id]['frame'])[-1] == list(negative['frame'])[-1]:
            continue
        elif len(negative[negative['particle']==particle_id]) > 20:
            pairs.append([list(negative[negative['particle']==particle_id]['frame'])[-1], list(negative[negative['particle']==particle_id]['x'])[-1], list(negative[negative['particle']==particle_id]['y'])[-1]])
        else:
            premature.append(particle_id)

    threshold_provided = 25
    # Find pairs close to each other and died in same frame 
    pairs_below_threshold_provided = []
    for pair in itertools.combinations(pairs, 2):
        distance = euclidean_distance(pair[0][1:], pair[1][1:]) 
        if distance < threshold_provided and pair[0][0]==pair[1][0]:
            pairs_below_threshold_provided.append(pair)
            distances.append(distance)
            density_distance.append(distance)
    density_distances.append(np.mean(density_distance))
    all_n_pairs.append(len(density_distance))
    all_premature.append(len(premature))
distances = np.array(distances)*8*5
density_distances = np.array(density_distances)*8*5

plt.hist(distances, bins=5, edgecolor = 'black')
plt.xlabel("Distance between singularities at pair annihilation ($\mathrm{\mu m}$)")
plt.ylabel("Density")
plt.axvline(distances.mean(), linestyle = 'dotted', c='black')
plt.show()

plt.scatter(date_densities.values(), density_distances, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Distance between singularities at \npair annihilation ($\mathrm{\mu m}$)")
plt.show()

plt.scatter(date_densities.values(), all_n_pairs, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of pair annihilation events")
plt.yticks([0, 5, 10, 15, 20])
plt.show()

plt.scatter(date_densities.values(), all_premature, c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of premature deaths")
plt.show()


#### MOTILITY-BASED PRUNING
plt.scatter(date_densities.values(), date_late_prune.values(), c='black')
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of motility-induced deaths")
z = np.polyfit(list(date_densities.values()), list(date_late_prune.values()), 1)
p = np.poly1d(z)
# plt.plot(list(date_densities.values()),p(list(date_densities.values())),"r--", alpha =0.4)
plt.show()
 

plt.scatter(date_densities.values(), final_number, label = "After premature and pair annihilations")
plt.scatter(date_densities.values(), date_aggregates.values(), alpha = 0.5, label = "Final aggregates")
plt.xlabel("Initial cell density ($cells/mm^2$)")
plt.ylabel("Number of singularities")
plt.legend()
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

for i in range(len(date_densities.values())):
    # if list(date_densities.values())[i] < 5000:
    #     low_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     low_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     low_motility.append(list(date_late_prune.values())[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    # elif list(date_densities.values())[i] < 6000:
    #     mid_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     mid_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     mid_motility.append(list(date_late_prune.values())[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))

    # else:
    #     high_premature.append(all_premature[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     high_pair.append(all_n_pairs[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))
    #     high_motility.append(list(date_late_prune.values())[i]/(all_premature[i]+all_n_pairs[i]+list(date_late_prune.values())[i]))

    if list(date_densities.values())[i] < 5000:
        low_premature.append(all_premature[i])
        low_pair.append(all_n_pairs[i])
        low_motility.append(list(date_late_prune.values())[i])
    elif list(date_densities.values())[i] < 6000:
        mid_premature.append(all_premature[i])
        mid_pair.append(all_n_pairs[i])
        mid_motility.append(list(date_late_prune.values())[i])

    else:
        high_premature.append(all_premature[i])
        high_pair.append(all_n_pairs[i])
        high_motility.append(list(date_late_prune.values())[i])


mean_low_premature = np.mean(low_premature)
mean_low_pair = np.mean(low_pair)
mean_low_motility = np.mean(low_motility)
mean_mid_premature = np.mean(mid_premature)
mean_mid_pair = np.mean(mid_pair)
mean_mid_motility = np.mean(mid_motility)
mean_high_premature = np.mean(high_premature)
mean_high_pair = np.mean(high_pair)
mean_high_motility = np.mean(high_motility)

low_cat = np.array([mean_low_premature, mean_low_pair, mean_low_motility])
low_cat/= low_cat.sum()
mid_cat = np.array([mean_mid_premature, mean_mid_pair, mean_mid_motility])
mid_cat/= mid_cat.sum()
high_cat = np.array([mean_high_premature, mean_high_pair, mean_high_motility])
high_cat/= high_cat.sum()

premature_cat=np.array([mean_low_premature, mean_mid_premature, mean_high_premature])
premature_cat/=premature_cat.sum()

pair_cat=np.array([mean_low_pair, mean_mid_pair, mean_high_pair])
pair_cat/=pair_cat.sum()
motility_cat=np.array([mean_low_motility, mean_mid_motility, mean_high_motility])
motility_cat /= motility_cat.sum()

# #### CONDITIONAL PROBABILITY ANALYSIS
# for i, date in enumerate(date_densities.keys()):
#     n_cells = date_densities[date]*120
#     optimal_number = n_cells/(10**5)
#     remainder = total_singularities[i]-optimal_number
#     c = all_premature[i]
#     p = all_n_pairs[i]
#     m = date_late_prune[date]
#     print(date, n_cells, total_singularities[i], optimal_number, remainder, c, p, m)

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

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Calculate the position of each bar
index = np.arange(3)
bar_width = 0.2
opacity = 0.8

rects1 = plt.bar(index - bar_width, [premature_cat[0], pair_cat[0], motility_cat[0]], bar_width, alpha=opacity, label='Low')
rects2 = plt.bar(index, [premature_cat[1], pair_cat[1], motility_cat[1]], bar_width, alpha=opacity, label='Mid')
rects3 = plt.bar(index + bar_width, [premature_cat[2], pair_cat[2], motility_cat[2]], bar_width, alpha=opacity, label='High')

plt.xlabel('Type of death', size = 12)
plt.ylabel('Mean proportion', size = 12)
plt.xticks(index, ('Premature', 'Pair annihilation', 'Motility-based'))
plt.legend()

plt.tight_layout()
plt.show()


# CREATION

# Plot histogram of singularity creation times
start_times = []
relative_start_times = []
super_sing_starts = []
for date in tqdm(date_densities):
    positive = pd.read_pickle(f"Data/{date}/Analysed_data/positive_df.pkl")
    negative = pd.read_pickle(f"Data/{date}/Analysed_data/negative_df.pkl")
    
    earliest_start_time = min(list(positive['frame'])[0], list(negative['frame'])[0])
    
    for i in positive['particle'].unique():
        start_times.append(list(positive[positive['particle']==i]['frame'])[0])
        relative_start_times.append(list(positive[positive['particle']==i]['frame'])[0]-earliest_start_time)
        if list(positive[positive['particle']==i]['frame'])[-1] ==max(positive['frame']):
            super_sing_starts.append(list(positive[positive['particle']==i]['frame'])[0])

    for i in negative['particle'].unique():
        start_times.append(list(negative[negative['particle']==i]['frame'])[0])
        relative_start_times.append(list(negative[negative['particle']==i]['frame'])[0]-earliest_start_time)
        if list(negative[negative['particle']==i]['frame'])[-1] ==max(negative['frame']):
            super_sing_starts.append(list(negative[negative['particle']==i]['frame'])[0])
relative_start_times = np.array(relative_start_times)
times = [3, 3.5, 4, 4.5, 5]
frames = (3600/15)*(np.array(times)-3)
plt.hist(start_times, density = True, bins = 30, label = "All singularities");
plt.hist(super_sing_starts, density = True, bins = 30, color='red', alpha =0.5, label = 'Supersingularities')
plt.xticks(frames, times);
plt.xlabel("Birth time (h)", fontsize = 12);
plt.ylabel("Probability density", fontsize = 12)
plt.legend()




#### FORMATION AND PRUNING RATES
all_creation_plots = []
all_death_plots = []
for date in date_densities:
    positive = pd.read_pickle(f"Data/{date}/Analysed_data/positive_df.pkl")
    negative = pd.read_pickle(f"Data/{date}/Analysed_data/negative_df.pkl")
    
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
    all_creation_plots.append(creation_rates)
    all_death_plots.append(death_rates)

for i, date in enumerate(date_densities):
    print(list(date_densities.values())[i])
    label_distribution=np.load(f"Data/{date}/Analysed_data/correlation_areas.npy")

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
    
