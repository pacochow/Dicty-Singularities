import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random

def create_colormap():
    cmap = plt.cm.twilight.copy()
    cmap.set_bad(color = 'black')

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
    return color_map