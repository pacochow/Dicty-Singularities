import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
from src.helpers import *
from src.analysis_utils import *

def create_data_timelapse(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(22,17) )
    
    a = images[0]

    im = plt.imshow(a, interpolation='none', cmap = 'gray', vmin = 110, vmax = 210)
    plt.axis('off')

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.95, f'{start_time//1:0>2}:{60*(start_time-(start_time//1))}:00', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=50)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        
        im.set_array(images[i])

        # Update iteration number text
        iteration_text.set_text(f'{int((start_time+frame_intervals*i/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*i/60)%60)//1):0>2}:{int(((frame_intervals*i)%60)//1):0>2}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = images.shape[0],
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')

def animate_processed_data(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, tracking = False):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    
    a = images[0]

    im = plt.imshow(a, cmap = create_colormap())

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize = 30)

    if tracking is not False:
        positive, negative = identify_singularities(tracking[0])
        plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c = 'black', marker = 's')
        plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker = 'o')



    plt.axis('off')

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.95, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}:{int(((frame_intervals*frame_start)%60)//1):0>2}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=50)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        
        im.set_array(images[i])
        
        # Remove previous scatter plots
        for collection in fig.gca().collections:
            collection.remove()
        # Scatter plot for singularities, if tracking is enabled
        if tracking is not False:

            positive, negative = identify_singularities(tracking[i])
            plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s', )
            plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')


        # Update iteration number text
        iteration_text.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}:{int(((frame_intervals*(frame_start+i))%60)//1):0>2}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = images.shape[0],
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')


def create_tracking_animation(phase: np.ndarray, tracking: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0):
    images = np.array([phase, tracking])
    iterations = images.shape[1]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot elements we want to animate
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
    
    
    # Titles
    titles = ["Phase", "Tracking singularities"]
    
    # Create an array to hold your image objects
    ims = []
    
    for j in range(2):  # loop over your new dimension
        a = images[j, 0]  # the initial state for each animation
        if j == 0:
            im = axs[j].imshow(a, cmap = 'hsv')
        else:
            im = axs[j].imshow(a, vmin = -1, vmax = 1)
        axs[j].axis('off')
        axs[j].set_title(titles[j], fontsize = 20)
        ims.append(im)
    
    # Add a centered global title
    global_title = fig.suptitle(f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}:{int(((frame_intervals*frame_start)%60)//1):0>2}', fontsize=40)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def animate_func(i):
        if i % fps == 0:
            print('.', end ='')
        
        for j in range(2):  # loop over your new dimension
            ims[j].set_array(images[j, i])  # update each animation

        # Update the global title with the current iteration number
        global_title.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}:{int(((frame_intervals*(frame_start+i))%60)//1):0>2}')

        return ims

    anim = animation.FuncAnimation(
        fig, 
        animate_func, 
        frames = iterations,
        interval = 1000 / fps, # in ms
        )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Animation done!')