import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random
from matplotlib.colors import LinearSegmentedColormap

def create_colormap():
    cmap = plt.cm.twilight.copy()
    cmap.set_bad(color = 'white')

    return cmap

def create_tracking_colormap(tracked_particles):
    # tracked_particles is dataframe

    # Create a list of colors from the tuple
    colors = list(plt.cm.tab20.colors)  # Convert tuple to list

    # Create a cyclic iterator of colors
    color_cycle = itertools.cycle(colors)

    # Map each unique particle to a color
    unique_particles = tracked_particles['particle'].unique()
    color_map = {particle: next(color_cycle) for particle in unique_particles}

    # Assign colors to each row in the dataframe
    tracked_particles['color'] = tracked_particles['particle'].map(color_map)

    return color_map, tracked_particles


def create_rainbow_colormap():

    # Define a sequence of colors for the gradient
    colors = ["#0000FF", "#32CD32", "#FFFF00", "#FFA500", "#FF0000"]  # Blue, Green, Yellow, Orange, Red

    # Create a custom colormap
    n_bins = 100  # Adjust for gradient smoothness
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=n_bins)

    return cmap


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
