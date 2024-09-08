import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks
import math
import cv2
import pandas as pd
from src.helpers import *

def classify_points_binarized(cells, index1, index2, box_size, time):
    range1 = (index1, index1+box_size)
    range2 = (index2, index2+box_size)

    raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)
    
    # Compute which pixels have cells
    binarized = np.apply_along_axis(cell_or_not, 0, raw_signal)

    # Set mask where there are no cells and compute mean 
    masked = np.where(binarized == 1, raw_signal, np.nan)

    # Get signal for box averaged over pixels with cells
    signal = np.nanmean(masked, axis = 1)
    
    # Smooth signal 
    smoothed_signal = savgol_filter(signal, 5, 2)

    # Compute derivatives
    derivatives = [(smoothed_signal[i+1]-smoothed_signal[i-1]) for i in range(1, len(smoothed_signal)-1)]

    # Find high peaks
    high_peaks = find_peaks(smoothed_signal, distance = 1, height = 1)[0]

    if len(high_peaks)==0:
        high_peaks = np.arange(time[0], time[1])

    # Linearly interpolate high peaks to compute floor
    floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])
    
    # Normalize signal
    diff = floor - smoothed_signal
    diff = savgol_filter(diff, 5, 2)

    # Compute threshold
    diff_peaks = find_peaks(diff, distance = 6, height = 1)[0]
    if len(diff_peaks)==0:
        diff_peaks = np.arange(time[0], time[1])
    diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
    threshold = savgol_filter(diff_floor, 50, 2)/2
    threshold = np.maximum(threshold, np.ones(len(threshold))*3)

    # Normalize diff
    normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]

    # Compute phase angles
    angles = np.array([(math.atan2(derivatives[i], normalized_signal[i])+np.pi)%(2*np.pi) for i in range(len(smoothed_signal)-2)])
    angles[signal[1:len(smoothed_signal)-1] == 0] = np.nan
    return angles

def classify_points_normalized(cells, index1, index2, box_size, time, base_threshold = 0.03):
    range1 = (index1, index1+box_size)
    range2 = (index2, index2+box_size)

    raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)
    
    # Compute which pixels have cells
    binarized = np.apply_along_axis(cell_or_not, 0, raw_signal)

    # Set mask where there are no cells and compute mean 
    masked = np.where(binarized == 1, raw_signal, np.nan)

    # Get signal for box averaged over pixels with cells
    signal = np.nanmean(masked, axis = 1)
    
    # Smooth signal 
    smoothed_signal = savgol_filter(signal, 5, 2)

    # Compute derivatives
    derivatives = [(smoothed_signal[i+1]-smoothed_signal[i-1]) for i in range(1, len(smoothed_signal)-1)]

    # Find high peaks
    high_peaks = find_peaks(smoothed_signal, distance = 20)[0]

    if len(high_peaks)==0:
        high_peaks = np.arange(time[0], time[1])

    # Linearly interpolate high peaks to compute floor
    floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])
    
    # Normalize signal
    diff = floor - smoothed_signal
    diff = savgol_filter(diff, 5, 2)

    # Compute threshold
    diff_peaks = find_peaks(diff, distance = 20)[0]
    if len(diff_peaks)==0:
        diff_peaks = np.arange(time[0], time[1])
    diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
    threshold = savgol_filter(diff_floor, 50, 2)/2
    threshold = np.maximum(threshold, np.ones(len(threshold))*base_threshold)
    # Normalize diff
    normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]


    # Compute phase angles
    angles = np.array([(math.atan2(derivatives[i], normalized_signal[i])+np.pi)%(2*np.pi) for i in range(len(smoothed_signal)-2)])
    angles[signal[1:len(smoothed_signal)-1] == 0] = np.nan
    return angles

def classify_points_time_normalized(cells, index1, index2, box_size, time, base_threshold = 0.03):
    range1 = (index1, index1+box_size)
    range2 = (index2, index2+box_size)

    raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)
    
    # Get signal for box averaged over pixels with cells
    signal = raw_signal.mean(axis=1)
    
    # Smooth signal 
    smoothed_signal = savgol_filter(signal, 10, 2)

 
    # Find high peaks
    high_peaks = find_peaks(smoothed_signal, distance = 20)[0]

    if len(high_peaks)==0:
        high_peaks = np.arange(time[0], time[1])

    # Linearly interpolate high peaks to compute floor
    floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])
    
    # Normalize signal
    diff = floor - smoothed_signal
    diff = savgol_filter(diff, 5, 2)

    # Compute derivatives
    derivatives = [(diff[i+1]-diff[i-1]) for i in range(1, len(diff)-1)]


    # Compute threshold
    diff_peaks = find_peaks(diff, distance = 20)[0]
    if len(diff_peaks)==0:
        diff_peaks = np.arange(time[0], time[1])
    diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
    threshold = savgol_filter(diff_floor, 50, 2)/2
    threshold = np.maximum(threshold, np.ones(len(threshold))*base_threshold)
    # Normalize diff
    normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]


    # Compute phase angles
    angles = np.array([(math.atan2(normalized_signal[i],derivatives[i])+np.pi/2)%(2*np.pi) for i in range(len(smoothed_signal)-2)])
    return angles


def classify_points_singularity_formation(cells, index1, index2, box_size, time, base_threshold = 1):
    range1 = (index1, index1+box_size)
    range2 = (index2, index2+box_size)

    raw_signal = cells[time[0]:time[1], range1[0]:range1[1], range2[0]:range2[1]].reshape(time[1]-time[0], (range1[1]-range1[0])**2)
    
    # Get signal for box averaged over pixels with cells
    signal = raw_signal.mean(axis=1)
    
    # Smooth signal 
    smoothed_signal = savgol_filter(signal, 35, 2)


    # Find high peaks
    high_peaks = find_peaks(smoothed_signal, distance = 30)[0]

    if len(high_peaks)==0:
        high_peaks = np.arange(time[0], time[1])

    # Linearly interpolate high peaks to compute floor
    floor = np.interp(np.arange(time[0], time[1]), high_peaks, smoothed_signal[high_peaks])
    
    # Normalize signal
    diff = floor - smoothed_signal
    diff = savgol_filter(diff, 5, 2)

    # Compute derivatives
    derivatives = [(diff[i+1]-diff[i-1]) for i in range(1, len(diff)-1)]


    # Compute threshold
    diff_peaks = find_peaks(diff, distance = 20)[0]
    if len(diff_peaks)==0:
        diff_peaks = np.arange(time[0], time[1])
    diff_floor = np.interp(np.arange(time[0], time[1]), diff_peaks, diff[diff_peaks])
    threshold = savgol_filter(diff_floor, 50, 2)/2
    threshold = np.maximum(threshold, np.ones(len(threshold))*base_threshold)
    # Normalize diff
    normalized_signal = (diff-threshold)[1:len(smoothed_signal)-1]


    # Compute phase angles
    angles = np.array([(math.atan2(normalized_signal[i],derivatives[i])+np.pi/2)%(2*np.pi) for i in range(len(smoothed_signal)-2)])
    return angles

def visualize_pixel_evolution(images, range1, range2, times):
    
    x1, x2 = range1
    y1, y2 = range2

    num_frames = times[1]-times[0]

    # Calculate number of rows and columns for subplots
    n_rows = int(np.sqrt(num_frames))
    n_cols = np.ceil(num_frames / n_rows).astype(int)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for i in range(num_frames):
        if i < len(images):
            # Extract the specified area
            area = images[i+times[0], x1:x2, y1:y2]
            axes[i].imshow(area, cmap='gray', vmin = 110, vmax = 210)
            axes[i].set_title(f"{i+1}", fontsize = 20)
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()









def phase_difference(phase1, phase2):
    diff = phase2 - phase1
    return np.arctan2(np.sin(diff), np.cos(diff))

def compute_winding_number(phase, box = 11):
    winding_numbers = np.zeros(phase.shape)
    for i in tqdm(range(box//2, phase.shape[1]-box//2)):
        for j in range(box//2, phase.shape[2]-box//2):
            eleven = np.array([
                phase_difference(phase[:, i-5, j-5], phase[:, i-4, j-5]),
                phase_difference(phase[:, i-4, j-5], phase[:, i-3, j-5]),
                phase_difference(phase[:, i-3, j-5], phase[:, i-2, j-5]),
                phase_difference(phase[:, i-2, j-5], phase[:, i-1, j-5]),
                phase_difference(phase[:, i-1, j-5], phase[:, i, j-5]),
                phase_difference(phase[:, i, j-5], phase[:, i+1, j-5]),
                phase_difference(phase[:, i+1, j-5], phase[:, i+2, j-5]),
                phase_difference(phase[:, i+2, j-5], phase[:, i+3, j-5]),
                phase_difference(phase[:, i+3, j-5], phase[:, i+4, j-5]),
                phase_difference(phase[:, i+4, j-5], phase[:, i+5, j-5]),
                phase_difference(phase[:, i+5,j-5], phase[:, i+5, j-4]),
                phase_difference(phase[:, i+5,j-4], phase[:, i+5, j-3]),
                phase_difference(phase[:, i+5,j-3], phase[:, i+5, j-2]),
                phase_difference(phase[:, i+5,j-2], phase[:, i+5, j-1]),
                phase_difference(phase[:, i+5,j-1], phase[:, i+5, j]),
                phase_difference(phase[:, i+5,j], phase[:, i+5, j+1]),
                phase_difference(phase[:, i+5,j+1], phase[:, i+5, j+2]),
                phase_difference(phase[:, i+5,j+2], phase[:, i+5, j+3]),
                phase_difference(phase[:, i+5,j+3], phase[:, i+5, j+4]),
                phase_difference(phase[:, i+5,j+4], phase[:, i+5, j+5]),
                phase_difference(phase[:, i+5, j+5], phase[:, i+4, j+5]),
                phase_difference(phase[:, i+4, j+5], phase[:, i+3, j+5]),
                phase_difference(phase[:, i+3, j+5], phase[:, i+2, j+5]),
                phase_difference(phase[:, i+2, j+5], phase[:, i+1, j+5]),
                phase_difference(phase[:, i+1, j+5], phase[:, i, j+5]),
                phase_difference(phase[:, i, j+5], phase[:, i-1, j+5]),
                phase_difference(phase[:, i-1, j+5], phase[:, i-2, j+5]),
                phase_difference(phase[:, i-2, j+5], phase[:, i-3, j+5]),
                phase_difference(phase[:, i-3, j+5], phase[:, i-4, j+5]),
                phase_difference(phase[:, i-4,j+5], phase[:, i-5, j+5]),
                phase_difference(phase[:, i-5,j+5], phase[:, i-5, j+4]),
                phase_difference(phase[:, i-5,j+4], phase[:, i-5, j+3]),
                phase_difference(phase[:, i-5,j+3], phase[:, i-5, j+2]),
                phase_difference(phase[:, i-5,j+2], phase[:, i-5, j+1]),
                phase_difference(phase[:, i-5,j+1], phase[:, i-5, j]),
                phase_difference(phase[:, i-5,j], phase[:, i-5, j-1]),
                phase_difference(phase[:, i-5,j-1], phase[:, i-5, j-2]),
                phase_difference(phase[:, i-5,j-2], phase[:, i-5, j-3]),
                phase_difference(phase[:, i-5,j-3], phase[:, i-5, j-4]),
                phase_difference(phase[:, i-5,j-4], phase[:, i-5, j-5])
            ])
            eleven_diffs = np.sum(eleven, axis = 0)

            nine = np.array([
                phase_difference(phase[:, i-4, j-4], phase[:, i-3, j-4]),
                phase_difference(phase[:, i-3, j-4], phase[:, i-2, j-4]),
                phase_difference(phase[:, i-2, j-4], phase[:, i-1, j-4]),
                phase_difference(phase[:, i-1, j-4], phase[:, i, j-4]),
                phase_difference(phase[:, i, j-4], phase[:, i+1, j-4]),
                phase_difference(phase[:, i+1, j-4], phase[:, i+2, j-4]),
                phase_difference(phase[:, i+2, j-4], phase[:, i+3, j-4]),
                phase_difference(phase[:, i+3, j-4], phase[:, i+4, j-4]),
                phase_difference(phase[:, i+4,j-4], phase[:, i+4, j-3]),
                phase_difference(phase[:, i+4,j-3], phase[:, i+4, j-2]),
                phase_difference(phase[:, i+4,j-2], phase[:, i+4, j-1]),
                phase_difference(phase[:, i+4,j-1], phase[:, i+4, j]),
                phase_difference(phase[:, i+4,j], phase[:, i+4, j+1]),
                phase_difference(phase[:, i+4,j+1], phase[:, i+4, j+2]),
                phase_difference(phase[:, i+4,j+2], phase[:, i+4, j+3]),
                phase_difference(phase[:, i+4,j+3], phase[:, i+4, j+4]),
                phase_difference(phase[:, i+4, j+4], phase[:, i+3, j+4]),
                phase_difference(phase[:, i+3, j+4], phase[:, i+2, j+4]),
                phase_difference(phase[:, i+2, j+4], phase[:, i+1, j+4]),
                phase_difference(phase[:, i+1, j+4], phase[:, i, j+4]),
                phase_difference(phase[:, i, j+4], phase[:, i-1, j+4]),
                phase_difference(phase[:, i-1, j+4], phase[:, i-2, j+4]),
                phase_difference(phase[:, i-2, j+4], phase[:, i-3, j+4]),
                phase_difference(phase[:, i-3, j+4], phase[:, i-4, j+4]),
                phase_difference(phase[:, i-4,j+4], phase[:, i-4, j+3]),
                phase_difference(phase[:, i-4,j+3], phase[:, i-4, j+2]),
                phase_difference(phase[:, i-4,j+2], phase[:, i-4, j+1]),
                phase_difference(phase[:, i-4,j+1], phase[:, i-4, j]),
                phase_difference(phase[:, i-4,j], phase[:, i-4, j-1]),
                phase_difference(phase[:, i-4,j-1], phase[:, i-4, j-2]),
                phase_difference(phase[:, i-4,j-2], phase[:, i-4, j-3]),
                phase_difference(phase[:, i-4,j-3], phase[:, i-4, j-4])
            ])

            nine_diffs = np.sum(nine, axis = 0)
            seven = np.array([
                    phase_difference(phase[:, i-3, j-3], phase[:, i-2, j-3]),
                    phase_difference(phase[:, i-2, j-3], phase[:, i-1, j-3]),
                    phase_difference(phase[:, i-1, j-3], phase[:, i, j-3]),
                    phase_difference(phase[:, i, j-3], phase[:, i+1, j-3]),
                    phase_difference(phase[:, i+1, j-3], phase[:, i+2, j-3]),
                    phase_difference(phase[:, i+2, j-3], phase[:, i+3, j-3]),
                    phase_difference(phase[:, i+3,j-3], phase[:, i+3, j-2]),
                    phase_difference(phase[:, i+3,j-2], phase[:, i+3, j-1]),
                    phase_difference(phase[:, i+3,j-1], phase[:, i+3, j]),
                    phase_difference(phase[:, i+3,j], phase[:, i+3, j+1]),
                    phase_difference(phase[:, i+3,j+1], phase[:, i+3, j+2]),
                    phase_difference(phase[:, i+3,j+2], phase[:, i+3, j+3]),
                    phase_difference(phase[:, i+3, j+3], phase[:, i+2, j+3]),
                    phase_difference(phase[:, i+2, j+3], phase[:, i+1, j+3]),
                    phase_difference(phase[:, i+1, j+3], phase[:, i, j+3]),
                    phase_difference(phase[:, i, j+3], phase[:, i-1, j+3]),
                    phase_difference(phase[:, i-1, j+3], phase[:, i-2, j+3]),
                    phase_difference(phase[:, i-2, j+3], phase[:, i-3, j+3]),
                    phase_difference(phase[:, i-3,j+3], phase[:, i-3, j+2]),
                    phase_difference(phase[:, i-3,j+2], phase[:, i-3, j+1]),
                    phase_difference(phase[:, i-3,j+1], phase[:, i-3, j]),
                    phase_difference(phase[:, i-3,j], phase[:, i-3, j-1]),
                    phase_difference(phase[:, i-3,j-1], phase[:, i-3, j-2]),
                    phase_difference(phase[:, i-3,j-2], phase[:, i-3, j-3])
                ])
            seven_diffs = np.sum(seven, axis =0)
            five = np.array([
                    phase_difference(phase[:, i-2, j-2], phase[:, i-1, j-2]),
                    phase_difference(phase[:, i-1, j-2], phase[:, i, j-2]),
                    phase_difference(phase[:, i, j-2], phase[:, i+1, j-2]),
                    phase_difference(phase[:, i+1, j-2], phase[:, i+2, j-2]),
                    phase_difference(phase[:, i+2,j-2], phase[:, i+2, j-1]),
                    phase_difference(phase[:, i+2,j-1], phase[:, i+2, j]),
                    phase_difference(phase[:, i+2,j], phase[:, i+2, j+1]),
                    phase_difference(phase[:, i+2,j+1], phase[:, i+2, j+2]),
                    phase_difference(phase[:, i+2, j+2], phase[:, i+1, j+2]),
                    phase_difference(phase[:, i+1, j+2], phase[:, i, j+2]),
                    phase_difference(phase[:, i, j+2], phase[:, i-1, j+2]),
                    phase_difference(phase[:, i-1, j+2], phase[:, i-2, j+2]),
                    phase_difference(phase[:, i-2,j+2], phase[:, i-2, j+1]),
                    phase_difference(phase[:, i-2,j+1], phase[:, i-2, j]),
                    phase_difference(phase[:, i-2,j], phase[:, i-2, j-1]),
                    phase_difference(phase[:, i-2,j-1], phase[:, i-2, j-2])
                ])
            five_diffs = np.sum(five, axis=0)
            three = np.array([
                    phase_difference(phase[:, i-1, j-1], phase[:, i, j-1]),
                    phase_difference(phase[:, i, j-1], phase[:, i+1, j-1]),
                    phase_difference(phase[:, i+1,j-1], phase[:, i+1, j]),
                    phase_difference(phase[:, i+1,j], phase[:, i+1, j+1]),
                    phase_difference(phase[:, i+1, j+1], phase[:, i, j+1]),
                    phase_difference(phase[:, i, j+1], phase[:, i-1, j+1]),
                    phase_difference(phase[:, i-1,j+1], phase[:, i-1, j]),
                    phase_difference(phase[:, i-1,j], phase[:, i-1, j-1])
                ])
            three_diffs = np.sum(three, axis=0)

            winding_numbers[:, i, j] = np.array([three_diffs, five_diffs, seven_diffs, nine_diffs, eleven_diffs]).mean(axis = 0)/(2*np.pi)
    return winding_numbers


def compute_winding_number_sf(phase, box = 15):
    winding_numbers = np.zeros(phase.shape)
    for i in tqdm(range(box//2, phase.shape[1]-box//2)):
        for j in range(box//2, phase.shape[2]-box//2):
            
            fifteen = np.array([
                phase_difference(phase[:, i-7, j-7], phase[:, i-6, j-7]),
                phase_difference(phase[:, i-6, j-7], phase[:, i-5, j-7]),
                phase_difference(phase[:, i-5, j-7], phase[:, i-4, j-7]),
                phase_difference(phase[:, i-4, j-7], phase[:, i-3, j-7]),
                phase_difference(phase[:, i-3, j-7], phase[:, i-2, j-7]),
                phase_difference(phase[:, i-2, j-7], phase[:, i-1, j-7]),
                phase_difference(phase[:, i-1, j-7], phase[:, i, j-7]),
                phase_difference(phase[:, i, j-7], phase[:, i+1, j-7]),
                phase_difference(phase[:, i+1, j-7], phase[:, i+2, j-7]),
                phase_difference(phase[:, i+2, j-7], phase[:, i+3, j-7]),
                phase_difference(phase[:, i+3, j-7], phase[:, i+4, j-7]),
                phase_difference(phase[:, i+4, j-7], phase[:, i+5, j-7]),
                phase_difference(phase[:, i+5, j-7], phase[:, i+6, j-7]),
                phase_difference(phase[:, i+6, j-7], phase[:, i+7, j-7]),
                phase_difference(phase[:, i+7, j-7], phase[:, i+7, j-6]),
                phase_difference(phase[:, i+7, j-6], phase[:, i+7, j-5]),
                phase_difference(phase[:, i+7,j-5], phase[:, i+7, j-4]),
                phase_difference(phase[:, i+7,j-4], phase[:, i+7, j-3]),
                phase_difference(phase[:, i+7,j-3], phase[:, i+7, j-2]),
                phase_difference(phase[:, i+7,j-2], phase[:, i+7, j-1]),
                phase_difference(phase[:, i+7,j-1], phase[:, i+7, j]),
                phase_difference(phase[:, i+7,j], phase[:, i+7, j+1]),
                phase_difference(phase[:, i+7,j+1], phase[:, i+7, j+2]),
                phase_difference(phase[:, i+7,j+2], phase[:, i+7, j+3]),
                phase_difference(phase[:, i+7,j+3], phase[:, i+7, j+4]),
                phase_difference(phase[:, i+7,j+4], phase[:, i+7, j+5]),
                phase_difference(phase[:, i+7,j+5], phase[:, i+7, j+6]),
                phase_difference(phase[:, i+7,j+6], phase[:, i+7, j+7]),
                phase_difference(phase[:, i+7, j+7], phase[:, i+6, j+7]),
                phase_difference(phase[:, i+6, j+7], phase[:, i+5, j+7]),
                phase_difference(phase[:, i+5, j+7], phase[:, i+4, j+7]),
                phase_difference(phase[:, i+4, j+7], phase[:, i+3, j+7]),
                phase_difference(phase[:, i+3, j+7], phase[:, i+2, j+7]),
                phase_difference(phase[:, i+2, j+7], phase[:, i+1, j+7]),
                phase_difference(phase[:, i+1, j+7], phase[:, i, j+7]),
                phase_difference(phase[:, i, j+7], phase[:, i-1, j+7]),
                phase_difference(phase[:, i-1, j+7], phase[:, i-2, j+7]),
                phase_difference(phase[:, i-2, j+7], phase[:, i-3, j+7]),
                phase_difference(phase[:, i-3, j+7], phase[:, i-4, j+7]),
                phase_difference(phase[:, i-4,j+7], phase[:, i-5, j+7]),
                phase_difference(phase[:, i-5,j+7], phase[:, i-6, j+7]),
                phase_difference(phase[:, i-6,j+7], phase[:, i-7, j+7]),
                phase_difference(phase[:, i-7,j+7], phase[:, i-7, j+6]),
                phase_difference(phase[:, i-7,j+6], phase[:, i-7, j+5]),
                phase_difference(phase[:, i-7,j+5], phase[:, i-7, j+4]),
                phase_difference(phase[:, i-7,j+4], phase[:, i-7, j+3]),
                phase_difference(phase[:, i-7,j+3], phase[:, i-7, j+2]),
                phase_difference(phase[:, i-7,j+2], phase[:, i-7, j+1]),
                phase_difference(phase[:, i-7,j+1], phase[:, i-7, j]),
                phase_difference(phase[:, i-7,j], phase[:, i-7, j-1]),
                phase_difference(phase[:, i-7,j-1], phase[:, i-7, j-2]),
                phase_difference(phase[:, i-7,j-2], phase[:, i-7, j-3]),
                phase_difference(phase[:, i-7,j-3], phase[:, i-7, j-4]),
                phase_difference(phase[:, i-7,j-4], phase[:, i-7, j-5]),
                phase_difference(phase[:, i-7,j-5], phase[:, i-7, j-6]),
                phase_difference(phase[:, i-7,j-6], phase[:, i-7, j-7])
            ])
            fifteen_diffs = np.sum(fifteen, axis = 0)

            thirteen = np.array([
                phase_difference(phase[:, i-6, j-6], phase[:, i-5, j-6]),
                phase_difference(phase[:, i-5, j-6], phase[:, i-4, j-6]),
                phase_difference(phase[:, i-4, j-6], phase[:, i-3, j-6]),
                phase_difference(phase[:, i-3, j-6], phase[:, i-2, j-6]),
                phase_difference(phase[:, i-2, j-6], phase[:, i-1, j-6]),
                phase_difference(phase[:, i-1, j-6], phase[:, i, j-6]),
                phase_difference(phase[:, i, j-6], phase[:, i+1, j-6]),
                phase_difference(phase[:, i+1, j-6], phase[:, i+2, j-6]),
                phase_difference(phase[:, i+2, j-6], phase[:, i+3, j-6]),
                phase_difference(phase[:, i+3, j-6], phase[:, i+4, j-6]),
                phase_difference(phase[:, i+4, j-6], phase[:, i+5, j-6]),
                phase_difference(phase[:, i+5, j-6], phase[:, i+6, j-6]),
                phase_difference(phase[:, i+6, j-6], phase[:, i+6, j-5]),
                phase_difference(phase[:, i+6,j-6], phase[:, i+6, j-4]),
                phase_difference(phase[:, i+6,j-4], phase[:, i+6, j-3]),
                phase_difference(phase[:, i+6,j-3], phase[:, i+6, j-2]),
                phase_difference(phase[:, i+6,j-2], phase[:, i+6, j-1]),
                phase_difference(phase[:, i+6,j-1], phase[:, i+6, j]),
                phase_difference(phase[:, i+6,j], phase[:, i+6, j+1]),
                phase_difference(phase[:, i+6,j+1], phase[:, i+6, j+2]),
                phase_difference(phase[:, i+6,j+2], phase[:, i+6, j+3]),
                phase_difference(phase[:, i+6,j+3], phase[:, i+6, j+4]),
                phase_difference(phase[:, i+6,j+4], phase[:, i+6, j+5]),
                phase_difference(phase[:, i+6,j+5], phase[:, i+6, j+6]),
                phase_difference(phase[:, i+6, j+6], phase[:, i+5, j+6]),
                phase_difference(phase[:, i+5, j+6], phase[:, i+4, j+6]),
                phase_difference(phase[:, i+4, j+6], phase[:, i+3, j+6]),
                phase_difference(phase[:, i+3, j+6], phase[:, i+2, j+6]),
                phase_difference(phase[:, i+2, j+6], phase[:, i+1, j+6]),
                phase_difference(phase[:, i+1, j+6], phase[:, i, j+6]),
                phase_difference(phase[:, i, j+6], phase[:, i-1, j+6]),
                phase_difference(phase[:, i-1, j+6], phase[:, i-2, j+6]),
                phase_difference(phase[:, i-2, j+6], phase[:, i-3, j+6]),
                phase_difference(phase[:, i-3, j+6], phase[:, i-4, j+6]),
                phase_difference(phase[:, i-4,j+6], phase[:, i-5, j+6]),
                phase_difference(phase[:, i-5,j+6], phase[:, i-6, j+6]),
                phase_difference(phase[:, i-6,j+6], phase[:, i-6, j+5]),
                phase_difference(phase[:, i-6,j+5], phase[:, i-6, j+4]),
                phase_difference(phase[:, i-6,j+4], phase[:, i-6, j+3]),
                phase_difference(phase[:, i-6,j+3], phase[:, i-6, j+2]),
                phase_difference(phase[:, i-6,j+2], phase[:, i-6, j+1]),
                phase_difference(phase[:, i-6,j+1], phase[:, i-6, j]),
                phase_difference(phase[:, i-6,j], phase[:, i-6, j-1]),
                phase_difference(phase[:, i-6,j-1], phase[:, i-6, j-2]),
                phase_difference(phase[:, i-6,j-2], phase[:, i-6, j-3]),
                phase_difference(phase[:, i-6,j-3], phase[:, i-6, j-4]),
                phase_difference(phase[:, i-6,j-4], phase[:, i-6, j-5]),
                phase_difference(phase[:, i-6,j-5], phase[:, i-6, j-6])
            ])
            thirteen_diffs = np.sum(thirteen, axis = 0)

            eleven = np.array([
                phase_difference(phase[:, i-5, j-5], phase[:, i-4, j-5]),
                phase_difference(phase[:, i-4, j-5], phase[:, i-3, j-5]),
                phase_difference(phase[:, i-3, j-5], phase[:, i-2, j-5]),
                phase_difference(phase[:, i-2, j-5], phase[:, i-1, j-5]),
                phase_difference(phase[:, i-1, j-5], phase[:, i, j-5]),
                phase_difference(phase[:, i, j-5], phase[:, i+1, j-5]),
                phase_difference(phase[:, i+1, j-5], phase[:, i+2, j-5]),
                phase_difference(phase[:, i+2, j-5], phase[:, i+3, j-5]),
                phase_difference(phase[:, i+3, j-5], phase[:, i+4, j-5]),
                phase_difference(phase[:, i+4, j-5], phase[:, i+5, j-5]),
                phase_difference(phase[:, i+5,j-5], phase[:, i+5, j-4]),
                phase_difference(phase[:, i+5,j-4], phase[:, i+5, j-3]),
                phase_difference(phase[:, i+5,j-3], phase[:, i+5, j-2]),
                phase_difference(phase[:, i+5,j-2], phase[:, i+5, j-1]),
                phase_difference(phase[:, i+5,j-1], phase[:, i+5, j]),
                phase_difference(phase[:, i+5,j], phase[:, i+5, j+1]),
                phase_difference(phase[:, i+5,j+1], phase[:, i+5, j+2]),
                phase_difference(phase[:, i+5,j+2], phase[:, i+5, j+3]),
                phase_difference(phase[:, i+5,j+3], phase[:, i+5, j+4]),
                phase_difference(phase[:, i+5,j+4], phase[:, i+5, j+5]),
                phase_difference(phase[:, i+5, j+5], phase[:, i+4, j+5]),
                phase_difference(phase[:, i+4, j+5], phase[:, i+3, j+5]),
                phase_difference(phase[:, i+3, j+5], phase[:, i+2, j+5]),
                phase_difference(phase[:, i+2, j+5], phase[:, i+1, j+5]),
                phase_difference(phase[:, i+1, j+5], phase[:, i, j+5]),
                phase_difference(phase[:, i, j+5], phase[:, i-1, j+5]),
                phase_difference(phase[:, i-1, j+5], phase[:, i-2, j+5]),
                phase_difference(phase[:, i-2, j+5], phase[:, i-3, j+5]),
                phase_difference(phase[:, i-3, j+5], phase[:, i-4, j+5]),
                phase_difference(phase[:, i-4,j+5], phase[:, i-5, j+5]),
                phase_difference(phase[:, i-5,j+5], phase[:, i-5, j+4]),
                phase_difference(phase[:, i-5,j+4], phase[:, i-5, j+3]),
                phase_difference(phase[:, i-5,j+3], phase[:, i-5, j+2]),
                phase_difference(phase[:, i-5,j+2], phase[:, i-5, j+1]),
                phase_difference(phase[:, i-5,j+1], phase[:, i-5, j]),
                phase_difference(phase[:, i-5,j], phase[:, i-5, j-1]),
                phase_difference(phase[:, i-5,j-1], phase[:, i-5, j-2]),
                phase_difference(phase[:, i-5,j-2], phase[:, i-5, j-3]),
                phase_difference(phase[:, i-5,j-3], phase[:, i-5, j-4]),
                phase_difference(phase[:, i-5,j-4], phase[:, i-5, j-5])
            ])
            eleven_diffs = np.sum(eleven, axis = 0)

            nine = np.array([
                phase_difference(phase[:, i-4, j-4], phase[:, i-3, j-4]),
                phase_difference(phase[:, i-3, j-4], phase[:, i-2, j-4]),
                phase_difference(phase[:, i-2, j-4], phase[:, i-1, j-4]),
                phase_difference(phase[:, i-1, j-4], phase[:, i, j-4]),
                phase_difference(phase[:, i, j-4], phase[:, i+1, j-4]),
                phase_difference(phase[:, i+1, j-4], phase[:, i+2, j-4]),
                phase_difference(phase[:, i+2, j-4], phase[:, i+3, j-4]),
                phase_difference(phase[:, i+3, j-4], phase[:, i+4, j-4]),
                phase_difference(phase[:, i+4,j-4], phase[:, i+4, j-3]),
                phase_difference(phase[:, i+4,j-3], phase[:, i+4, j-2]),
                phase_difference(phase[:, i+4,j-2], phase[:, i+4, j-1]),
                phase_difference(phase[:, i+4,j-1], phase[:, i+4, j]),
                phase_difference(phase[:, i+4,j], phase[:, i+4, j+1]),
                phase_difference(phase[:, i+4,j+1], phase[:, i+4, j+2]),
                phase_difference(phase[:, i+4,j+2], phase[:, i+4, j+3]),
                phase_difference(phase[:, i+4,j+3], phase[:, i+4, j+4]),
                phase_difference(phase[:, i+4, j+4], phase[:, i+3, j+4]),
                phase_difference(phase[:, i+3, j+4], phase[:, i+2, j+4]),
                phase_difference(phase[:, i+2, j+4], phase[:, i+1, j+4]),
                phase_difference(phase[:, i+1, j+4], phase[:, i, j+4]),
                phase_difference(phase[:, i, j+4], phase[:, i-1, j+4]),
                phase_difference(phase[:, i-1, j+4], phase[:, i-2, j+4]),
                phase_difference(phase[:, i-2, j+4], phase[:, i-3, j+4]),
                phase_difference(phase[:, i-3, j+4], phase[:, i-4, j+4]),
                phase_difference(phase[:, i-4,j+4], phase[:, i-4, j+3]),
                phase_difference(phase[:, i-4,j+3], phase[:, i-4, j+2]),
                phase_difference(phase[:, i-4,j+2], phase[:, i-4, j+1]),
                phase_difference(phase[:, i-4,j+1], phase[:, i-4, j]),
                phase_difference(phase[:, i-4,j], phase[:, i-4, j-1]),
                phase_difference(phase[:, i-4,j-1], phase[:, i-4, j-2]),
                phase_difference(phase[:, i-4,j-2], phase[:, i-4, j-3]),
                phase_difference(phase[:, i-4,j-3], phase[:, i-4, j-4])
            ])

            nine_diffs = np.sum(nine, axis = 0)
            seven = np.array([
                    phase_difference(phase[:, i-3, j-3], phase[:, i-2, j-3]),
                    phase_difference(phase[:, i-2, j-3], phase[:, i-1, j-3]),
                    phase_difference(phase[:, i-1, j-3], phase[:, i, j-3]),
                    phase_difference(phase[:, i, j-3], phase[:, i+1, j-3]),
                    phase_difference(phase[:, i+1, j-3], phase[:, i+2, j-3]),
                    phase_difference(phase[:, i+2, j-3], phase[:, i+3, j-3]),
                    phase_difference(phase[:, i+3,j-3], phase[:, i+3, j-2]),
                    phase_difference(phase[:, i+3,j-2], phase[:, i+3, j-1]),
                    phase_difference(phase[:, i+3,j-1], phase[:, i+3, j]),
                    phase_difference(phase[:, i+3,j], phase[:, i+3, j+1]),
                    phase_difference(phase[:, i+3,j+1], phase[:, i+3, j+2]),
                    phase_difference(phase[:, i+3,j+2], phase[:, i+3, j+3]),
                    phase_difference(phase[:, i+3, j+3], phase[:, i+2, j+3]),
                    phase_difference(phase[:, i+2, j+3], phase[:, i+1, j+3]),
                    phase_difference(phase[:, i+1, j+3], phase[:, i, j+3]),
                    phase_difference(phase[:, i, j+3], phase[:, i-1, j+3]),
                    phase_difference(phase[:, i-1, j+3], phase[:, i-2, j+3]),
                    phase_difference(phase[:, i-2, j+3], phase[:, i-3, j+3]),
                    phase_difference(phase[:, i-3,j+3], phase[:, i-3, j+2]),
                    phase_difference(phase[:, i-3,j+2], phase[:, i-3, j+1]),
                    phase_difference(phase[:, i-3,j+1], phase[:, i-3, j]),
                    phase_difference(phase[:, i-3,j], phase[:, i-3, j-1]),
                    phase_difference(phase[:, i-3,j-1], phase[:, i-3, j-2]),
                    phase_difference(phase[:, i-3,j-2], phase[:, i-3, j-3])
                ])
            seven_diffs = np.sum(seven, axis =0)
            five = np.array([
                    phase_difference(phase[:, i-2, j-2], phase[:, i-1, j-2]),
                    phase_difference(phase[:, i-1, j-2], phase[:, i, j-2]),
                    phase_difference(phase[:, i, j-2], phase[:, i+1, j-2]),
                    phase_difference(phase[:, i+1, j-2], phase[:, i+2, j-2]),
                    phase_difference(phase[:, i+2,j-2], phase[:, i+2, j-1]),
                    phase_difference(phase[:, i+2,j-1], phase[:, i+2, j]),
                    phase_difference(phase[:, i+2,j], phase[:, i+2, j+1]),
                    phase_difference(phase[:, i+2,j+1], phase[:, i+2, j+2]),
                    phase_difference(phase[:, i+2, j+2], phase[:, i+1, j+2]),
                    phase_difference(phase[:, i+1, j+2], phase[:, i, j+2]),
                    phase_difference(phase[:, i, j+2], phase[:, i-1, j+2]),
                    phase_difference(phase[:, i-1, j+2], phase[:, i-2, j+2]),
                    phase_difference(phase[:, i-2,j+2], phase[:, i-2, j+1]),
                    phase_difference(phase[:, i-2,j+1], phase[:, i-2, j]),
                    phase_difference(phase[:, i-2,j], phase[:, i-2, j-1]),
                    phase_difference(phase[:, i-2,j-1], phase[:, i-2, j-2])
                ])
            five_diffs = np.sum(five, axis=0)
            three = np.array([
                    phase_difference(phase[:, i-1, j-1], phase[:, i, j-1]),
                    phase_difference(phase[:, i, j-1], phase[:, i+1, j-1]),
                    phase_difference(phase[:, i+1,j-1], phase[:, i+1, j]),
                    phase_difference(phase[:, i+1,j], phase[:, i+1, j+1]),
                    phase_difference(phase[:, i+1, j+1], phase[:, i, j+1]),
                    phase_difference(phase[:, i, j+1], phase[:, i-1, j+1]),
                    phase_difference(phase[:, i-1,j+1], phase[:, i-1, j]),
                    phase_difference(phase[:, i-1,j], phase[:, i-1, j-1])
                ])
            three_diffs = np.sum(three, axis=0)

            winding_numbers[:, i, j] = np.array([three_diffs, five_diffs, seven_diffs, nine_diffs, eleven_diffs, thirteen_diffs, fifteen_diffs]).mean(axis = 0)/(2*np.pi)
    return winding_numbers

def identify_singularities(winding_numbers: np.ndarray, box = 11):
    positive_singularities = np.where(winding_numbers > 0.9)
    negative_singularities = np.where(winding_numbers < -0.9)  
    
    # Remove edge singularities
    positive = [
        (positive_singularities[0][i], positive_singularities[1][i]) for i in range(positive_singularities[0].shape[0]) if 
        positive_singularities[0][i]>=box//2 and 
        positive_singularities[0][i]<=winding_numbers.shape[-2]-box//2 and 
        positive_singularities[1][i]>=box//2 and 
        positive_singularities[1][i]<=winding_numbers.shape[-1]-box//2]
    negative = [
        (negative_singularities[0][i], negative_singularities[1][i]) for i in range(negative_singularities[0].shape[0]) if 
        negative_singularities[0][i]>=box//2 and 
        negative_singularities[0][i]<=winding_numbers.shape[-2]-box//2 and 
        negative_singularities[1][i]>=box//2 and 
        negative_singularities[1][i]<=winding_numbers.shape[-1]-box//2]
    return positive, negative
    

def phase_mean_pooling(phase_data, pool_size=(2, 2)):
    # Convert to complex representation
    complex_data = np.exp(1j * phase_data)
 
    # Reshape and pool
    sh = phase_data.shape[0], phase_data.shape[1]//pool_size[0], pool_size[0], phase_data.shape[2]//pool_size[1], pool_size[1]
    pooled_complex = np.mean(complex_data.reshape(sh), axis=(-1, -3))

    # Convert back to phase
    pooled_phase = np.angle(pooled_complex)

    return pooled_phase

def phase_gaussian_smoothing(phase_data, sigma = 1):
    # Convert to complex representation
    complex_representation = np.exp(1j * phase_data)

    # Apply Gaussian filter to the real and imaginary parts separately
    filtered_real = gaussian_filter(np.real(complex_representation), sigma=sigma, axes = (1, 2))
    filtered_imag = gaussian_filter(np.imag(complex_representation), sigma=sigma, axes=(1, 2))

    # Combine the filtered real and imaginary parts
    filtered_complex = filtered_real + 1j * filtered_imag

    # Convert back to phase
    filtered_phase = np.angle(filtered_complex)

    return filtered_phase

def get_edges(phase_data):

    scaled = (phase_data - phase_data.min()) / (phase_data.max() - phase_data.min()) * 255

    # Convert to uint8
    converted = scaled.astype(np.uint8)
    edges = np.zeros_like(phase_data)

    for t in range(edges.shape[0]):
        frame = converted[t]
        edges[t] = cv2.Canny(frame, 244, 250)
    return edges

def remove_zeros(signal):
    mask = signal==0
    padded = np.pad(signal, 1, mode = 'constant', constant_values=0)
    neighbor_means = (padded[:-2] + padded[2:]) / 2
    signal[mask] = neighbor_means[mask]
    return signal


def cell_or_not(signal, threshold = 110):
    binarized = signal>threshold
    convolve = np.convolve(np.ones(20), binarized, mode = 'same')
    binarized_convolve = convolve > 0
    return binarized_convolve


def cell_or_not_normalized(signal, threshold = 0.6):
    binarized = signal>threshold
    convolve = np.convolve(np.ones(6), binarized, mode = 'same')
    binarized_convolve = convolve > 0
    return binarized_convolve


def get_n_singularities(data, total_frames = 900):

    singularities = pd.read_pickle(f"Data/{data}/Analysed_data/singularities.pkl")

    n_singularities = np.array([len(singularities[singularities['frame']==i]) for i in range(total_frames)])
    return n_singularities

def smooth_singularity_time_series(n_singularities, since_birth = False):
    running_mean = pd.Series(n_singularities).rolling(window=5).mean()
    running_mean = savgol_filter(running_mean, 80, 2)
    if since_birth==True:
        running_mean = running_mean[running_mean>0.5]
    return running_mean

def get_n_singularities_since_birth(data, total_frames = 900):
    positive = pd.read_pickle(f"Data/{data}/Analysed_data/positive_df.pkl")
    negative = pd.read_pickle(f"Data/{data}/Analysed_data/negative_df.pkl")

    _, positive = create_tracking_colormap(positive)
    _, negative = create_tracking_colormap(negative)

    n_singularities = np.array([len(positive[positive['frame']==i])+len(negative[negative['frame']==i]) for i in range(total_frames)])
    running_mean = pd.Series(n_singularities).rolling(window=5).mean()
    running_mean = savgol_filter(running_mean, 80, 2)
    n_singularities = n_singularities[n_singularities!=0]
    running_mean = running_mean[running_mean>0.5]
    return n_singularities


def filter_singularities(t1_p, t1_n, pre_distance, post_distance, frame_cutoff_hard, frame_cutoff_side, stop_singularity_tracking, length = 275, width = 207):

    """
    Pre_distance and post_distance: [Left, right, top, bottom]
    Frame_cutoff_hard: No new singularities formed after this
    Frame_cutoff_side: No new singularities formed that moves to the side after this
    Stop_singularity_tracking: No more singularities tracked after this
    """

    t1_p=t1_p[t1_p['frame']<stop_singularity_tracking]
    t1_n=t1_n[t1_n['frame']<stop_singularity_tracking]

    # Filter by speed, position and frame
    for i in t1_p['particle'].unique():
        # Select the subset of the DataFrame for the current particle
        particle_data = t1_p[t1_p['particle'] == i]

        # Extract x and y coordinates
        xs = np.array(particle_data['x'].tolist())
        ys = np.array(particle_data['y'].tolist())
        times = particle_data['frame'].tolist()

        # Calculate the distance
        dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        speed = dist/(times[-1]-times[0])
        first_frame = particle_data['frame'].to_list()[0]

        if first_frame > frame_cutoff_hard or speed>2:
            # Drop rows where 'particle' column is equal to i
            t1_p = t1_p[t1_p['particle'] != i]
            
        elif len(xs) < 100 and (xs[-1]<pre_distance[0] or xs[-1]>width-pre_distance[1] or ys[-1]<pre_distance[2] or ys[-1]>length-pre_distance[3]):
            t1_p = t1_p[t1_p['particle'] != i]
        elif (first_frame > frame_cutoff_side and np.any(xs<post_distance[0])) or (first_frame > frame_cutoff_side and np.any(xs>width-post_distance[1])) or (first_frame > frame_cutoff_side and np.any(ys<post_distance[2])) or (first_frame>frame_cutoff_side and np.any(ys>length-post_distance[3])):

            t1_p = t1_p[t1_p['particle'] != i]

    for i in t1_n['particle'].unique():
        # Select the subset of the DataFrame for the current particle
        particle_data = t1_n[t1_n['particle'] == i]

        # Extract x and y coordinates
        xs = np.array(particle_data['x'].tolist())
        ys = np.array(particle_data['y'].tolist())
        times = particle_data['frame'].tolist()
        # Calculate the distance
        dist = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        speed = dist/(times[-1]-times[0])
        first_frame = particle_data['frame'].to_list()[0]

        if first_frame > frame_cutoff_hard or speed >2:
            t1_n = t1_n[t1_n['particle'] != i]
        elif len(xs) < 100 and (xs[-1]<pre_distance[0] or xs[-1]>width-pre_distance[1] or ys[-1]<pre_distance[2] or ys[-1]>length-pre_distance[3]):
            t1_n = t1_n[t1_n['particle'] != i]
        elif (first_frame > frame_cutoff_side and np.any(xs<post_distance[0])) or (first_frame > frame_cutoff_side and np.any(xs>width-post_distance[1])) or (first_frame > frame_cutoff_side and np.any(ys<post_distance[2])) or (first_frame>frame_cutoff_side and np.any(ys>length-post_distance[3])):
            t1_n = t1_n[t1_n['particle'] != i]

    return t1_p, t1_n




def fill_gaps(group):
    # Function to fill gaps for a single particle
    frames = group['frame'].values
    missing_frames = []
    for i in range(len(frames) - 1):
        current_frame, next_frame = frames[i], frames[i + 1]
        gap_frames = list(range(current_frame + 1, next_frame))
        if gap_frames:
            missing_data = {'particle': group['particle'].iloc[0], 'frame': gap_frames,
                            'x': [group['x'].iloc[i]] * len(gap_frames), 'y': [group['y'].iloc[i]] * len(gap_frames)}
            missing_df = pd.DataFrame(missing_data)
            missing_frames.append(missing_df)
    if missing_frames:
        return pd.concat(missing_frames, ignore_index=True)
    else:
        return pd.DataFrame()