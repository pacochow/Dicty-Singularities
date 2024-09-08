import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
from src.helpers import *
from src.analysis_utils import *
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

def create_data_timelapse(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(22,17) )
    
    a = images[0]

    # im = plt.imshow(a, interpolation='none', cmap = 'gray', vmin = 110, vmax = 210)
    im = plt.imshow(a, interpolation='none', cmap = 'gray', vmin = 0.7, vmax = 1.5)
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

def animate_processed_data(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, identifying = False, tracking = False, coordinates = False):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    
    a = images[0]

    im = plt.imshow(a, cmap = create_colormap(), vmin = -np.pi, vmax = np.pi)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize = 30)

    if identifying is not False:
        positive, negative = identify_singularities(identifying[0])
        plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c = 'black', marker = 's')
        plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker = 'o')

    if tracking is not False:
        positive, negative = tracking
        colormap_p, _ = create_tracking_colormap(positive)
        colormap_n, _ = create_tracking_colormap(negative)

        for particle in positive[positive['frame'] == 0]['particle']:
            particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
        for particle in negative[negative['frame'] == 0]['particle']:
            particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]  
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')

    if coordinates == False:
        plt.axis('off')
    else:
        plt.xticks(np.arange(0, images.shape[2], 10))
        plt.yticks(np.arange(0, images.shape[1], 10))

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.9, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}:{int(((frame_intervals*frame_start)%60)//1):0>2}\nFrame: {frame_start}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=40)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(images[i])
        
        # Remove previous scatter plots
        for collection in fig.gca().collections:
            collection.remove()
        # Scatter plot for singularities, if tracking is enabled
        if identifying is not False:

            positive, negative = identify_singularities(identifying[i])
            plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s', )
            plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

        if tracking is not False:
            positive, negative = tracking
            colormap_p, _ = create_tracking_colormap(positive)
            colormap_n, _ = create_tracking_colormap(negative)
            for particle in positive[positive['frame'] == i]['particle']:
                particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
            for particle in negative[negative['frame'] == i]['particle']:
                particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')
                



        # Update iteration number text
        iteration_text.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}:{int(((frame_intervals*(frame_start+i))%60)//1):0>2}\nFrame: {frame_start+i}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = images.shape[0],
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')

def animate_processed_data_fig(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, identifying = False, tracking = False, coordinates = False):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    a = images[0]

    im = plt.imshow(a, cmap = create_colormap(), vmin = -np.pi, vmax = np.pi)

    # cbar = plt.colorbar(im)
    # cbar.ax.tick_params(labelsize = 30)

    if identifying is not False:
        positive, negative = identify_singularities(identifying[0])
        plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c = 'black', marker = 's')
        plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker = 'o')

    if tracking is not False:
        positive, negative = tracking
        colormap_p, _ = create_tracking_colormap(positive)
        colormap_n, _ = create_tracking_colormap(negative)

        for particle in positive[positive['frame'] == 0]['particle']:
            particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
        for particle in negative[negative['frame'] == 0]['particle']:
            particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]  
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')

    if coordinates == False:
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    else:
        plt.xticks(np.arange(0, images.shape[2], 10))
        plt.yticks(np.arange(0, images.shape[1], 10))
    plt.xlim([5, 195])
    plt.ylim([265, 10])
    ax=fig.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(15)
    
    # ax=fig.gca()
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(10)

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.9, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=40, color = 'black')

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(images[i])
        
        # Remove previous scatter plots
        for collection in fig.gca().collections:
            collection.remove()
        # Scatter plot for singularities, if tracking is enabled
        if identifying is not False:

            positive, negative = identify_singularities(identifying[i])
            plt.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s', )
            plt.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

        if tracking is not False:
            positive, negative = tracking
            colormap_p, _ = create_tracking_colormap(positive)
            colormap_n, _ = create_tracking_colormap(negative)
            for particle in positive[positive['frame'] == i]['particle']:
                particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
            for particle in negative[negative['frame'] == i]['particle']:
                particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')


        # Update iteration number text
        iteration_text.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = images.shape[0],
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')


def animate_processed_data_with_raw_fig(raw: np.ndarray, processed: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, identifying = False, tracking = False, coordinates = False):
    iterations = raw.shape[0]
    fps = iterations / nSecs

    # Set up the figure, the axis, and the plot elements we want to animate
    fig, axes = plt.subplots(1, 2, figsize=(30, 17))
    fig.subplots_adjust(wspace=0.2)  # Adjust this value to reduce whitespace

    # Initial images
    im1 = axes[0].imshow(raw[0], cmap='gray', vmin = 0.8, vmax = 1.2)
    im2 = axes[1].imshow(processed[0], cmap=create_colormap(), vmin=-np.pi, vmax=np.pi)
    
    # Setup for axes[1]
    if identifying is not False:
        positive, negative = identify_singularities(identifying[0])
        axes[1].scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s')
        axes[1].scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

    if tracking is not False:
        positive, negative = tracking
        colormap_p, _ = create_tracking_colormap(positive)
        colormap_n, _ = create_tracking_colormap(negative)
        for particle in positive[positive['frame'] == 0]['particle']:
            particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
            axes[1].scatter(particle_data['x'], particle_data['y'], s=400, color=colormap_p[particle], marker='s')
        for particle in negative[negative['frame'] == 0]['particle']:
            particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]
            axes[1].scatter(particle_data['x'], particle_data['y'], s=400, color=colormap_n[particle], marker='o')

    # Coordinates handling
    for ax in axes:
        if coordinates == False:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks(np.arange(0, raw.shape[2], 10))
            ax.set_yticks(np.arange(0, raw.shape[1], 10))
        
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(15)
    axes[0].set_xlim([10, 390])
    axes[0].set_ylim([530, 20])
    axes[1].set_xlim([5, 195])
    axes[1].set_ylim([265, 10])
    # plt.tight_layout()
    # Create text elements to display iteration number
    iteration_text = fig.text(0.51, 0.95, f'{int((start_time + frame_intervals * frame_start / 3600) // 1):0>2}:{int(((60 * (start_time - (start_time // 1)) + frame_intervals * frame_start / 60) % 60) // 1):0>2}', ha='center', fontsize=40, color='black')

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        im1.set_array(raw[i])
        im2.set_array(processed[i])

        # Remove previous scatter plots
        for ax in axes:
            for collection in ax.collections:
                collection.remove()

        # Update scatter plots for images2
        if identifying is not False:
            positive, negative = identify_singularities(identifying[i])
            axes[1].scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s')
            axes[1].scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

        if tracking is not False:
            positive, negative = tracking
            colormap_p, _ = create_tracking_colormap(positive)
            colormap_n, _ = create_tracking_colormap(negative)
            for particle in positive[positive['frame'] == i]['particle']:
                particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                axes[1].scatter(particle_data['x'], particle_data['y'], s=400, color=colormap_p[particle], marker='s')
            for particle in negative[negative['frame'] == i]['particle']:
                particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]
                axes[1].scatter(particle_data['x'], particle_data['y'], s=400, color=colormap_n[particle], marker='o')

        # Update iteration number text
        iteration_text.set_text(f'{int((start_time + frame_intervals * (frame_start + i) / 3600) // 1):0>2}:{int(((60 * (start_time - (start_time // 1)) + frame_intervals * (frame_start + i) / 60) % 60) // 1):0>2}')

        return [im1, im2, iteration_text]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=raw.shape[0],
        interval=1000 / fps,  # in ms
    )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Full run done!')



def animate_processed_data_many_fig(images_list: list, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, identifying = False, tracking = False, coordinates = False):

    assert len(images_list) == 6, "You must provide exactly 6 image arrays."
    
    iterations = images_list[0].shape[0]
    fps = iterations/nSecs

    # Set up the figure with 6 subplots in one line
    fig, axs = plt.subplots(1, 6, figsize=(30, 7))
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0)
    texts=["2000-3000", "3000-4000", "4000-5000", "5000-6000", "6000-7000", "7000-8000"]
    ims = []
    for k, ax in enumerate(axs):
        a = images_list[k][0]
        im = ax.imshow(a, cmap=create_colormap(), vmin=-np.pi, vmax=np.pi)
        ims.append(im)
        
        # Add text at the top of each animation
        ax.text(0.5, 1.01, texts[k], transform=ax.transAxes, ha='center', va='bottom', fontsize=20, color='black')

        if identifying is not False:
            positive, negative = identify_singularities(identifying[k][0])
            ax.scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s')
            ax.scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

        if tracking is not False:
            positive, negative = tracking[k]
            colormap_p, _ = create_tracking_colormap(positive)
            colormap_n, _ = create_tracking_colormap(negative)

            for particle in positive[positive['frame'] == 0]['particle']:
                particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
                ax.scatter(particle_data['x'], particle_data['y'], s=100, color=colormap_p[particle], marker='s')
            for particle in negative[negative['frame'] == 0]['particle']:
                particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]  
                ax.scatter(particle_data['x'], particle_data['y'], s=100, color=colormap_n[particle], marker='o')

        if coordinates == False:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks(np.arange(0, images_list[k].shape[2], 10))
            ax.set_yticks(np.arange(0, images_list[k].shape[1], 10))
        ax.set_xlim([5, 195])
        ax.set_ylim([265, 10])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)

    iteration_text = fig.text(0.51, 0.93, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}', 
                               ha='center', fontsize=30, color='black')

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        for j, im in enumerate(ims):
            im.set_array(images_list[j][i])
            
            # Remove previous scatter plots
            for collection in axs[j].collections:
                collection.remove()
                
            if identifying is not False:
                positive, negative = identify_singularities(identifying[j][i])
                axs[j].scatter([x[1] for x in positive], [x[0] for x in positive], s=500, c='black', marker='s')
                axs[j].scatter([x[1] for x in negative], [x[0] for x in negative], s=500, c='white', marker='o')

            if tracking is not False:
                positive, negative = tracking[j]
                colormap_p, _ = create_tracking_colormap(positive)
                colormap_n, _ = create_tracking_colormap(negative)
                for particle in positive[positive['frame'] == i]['particle']:
                    particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                    axs[j].scatter(particle_data['x'], particle_data['y'], s=100, color=colormap_p[particle], marker='s')
                for particle in negative[negative['frame'] == i]['particle']:
                    particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]
                    axs[j].scatter(particle_data['x'], particle_data['y'], s=100, color=colormap_n[particle], marker='o')

        iteration_text.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}')

        return ims + [iteration_text]

    anim = animation.FuncAnimation(fig, 
                                   animate_func, 
                                   frames=iterations,
                                   interval=1000 / fps)
    
    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
    print('Full run done!')

def animate_for_aesthetics(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start=0):
    iterations = images.shape[0]
    fps = iterations / nSecs

    # Dynamically adjust figsize based on the images' aspect ratio to slightly reduce or add white space
    img_height, img_width = images[0].shape[:2]
    fig_aspect_ratio = img_width / img_height
    fig_width = 10  # Set figure width to a fixed size; adjust as needed
    fig_height = fig_width / fig_aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Adjust margins here to add a tiny bit of white space
    margin_size = 0.05  # Adjust the margin size as needed, 0.01 is just an example for minimal space
    plt.subplots_adjust(left=margin_size, right=1-margin_size, bottom=margin_size, top=1-margin_size, wspace=0, hspace=0)

    a = images[0]
    im = ax.imshow(a, cmap='hsv', vmin=-np.pi, vmax=np.pi)

    ax.axis('off')  # Hide the axes

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        im.set_data(images[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=iterations,
        interval=1000 / fps,  # Frame update interval
    )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Full run done!')


def animate_edges(images: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, frame_start = 0, periodicities = False):

    iterations = images.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    
    a = images[0]

    im = plt.imshow(a, cmap = 'gray', vmin = 0, vmax = 255)

    if periodicities != False:
        positive, negative = periodicities


        # Determine the range of frequencies for colorbar
        all_pers = np.concatenate([positive['period'], negative['period']])
        min_freq, max_freq = np.min(all_pers), np.quantile(all_pers, [0.95])[0]

        # Create a ScalarMappable for the colorbar
        norm = Normalize(vmin=min_freq, vmax=max_freq)
        sm = ScalarMappable(norm=norm, cmap='gist_rainbow')

        # Add a colorbar to the figure
        cbar = plt.colorbar(sm, ax=fig.gca(), orientation='vertical')
        cbar.set_label('Period (mins)', size=30)
        cbar.ax.tick_params(labelsize = 30)
        

        for particle in positive[positive['frame'] == 0]['particle']:
            particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=plt.cm.gist_rainbow(norm(particle_data['period'])), marker = 's')
        for particle in negative[negative['frame'] == 0]['particle']:
            particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]  
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=plt.cm.gist_rainbow(norm(particle_data['period'])), marker = 'o')


    plt.axis('off')

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.95, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}:{int(((frame_intervals*frame_start)%60)//1):0>2}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=50)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        
        im.set_array(images[i])

        if periodicities !=False:
            # Remove previous scatter plots
            for collection in fig.gca().collections:
                collection.remove()
            for particle in positive[positive['frame'] == i]['particle']:
                particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color = plt.cm.gist_rainbow(norm(particle_data['period'])), marker = 's')
            for particle in negative[negative['frame'] == i]['particle']:
                particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]  
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=plt.cm.gist_rainbow(norm(particle_data['period'])), marker = 'o')


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


def animate_periods(periods: np.ndarray, filename: str, nSecs: float, start_time: float, frame_intervals: float, tracking, frame_start = 0):

    iterations = periods.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    
    a = periods[0]

    im = plt.imshow(a, vmin = 0, vmax = 10)


    if tracking is not False:
        positive, negative = tracking
        colormap_p, _ = create_tracking_colormap(positive)
        colormap_n, _ = create_tracking_colormap(negative)

        for particle in positive[positive['frame'] == 0]['particle']:
            particle_data = positive[(positive['frame'] == 0) & (positive['particle'] == particle)]
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
        for particle in negative[negative['frame'] == 0]['particle']:
            particle_data = negative[(negative['frame'] == 0) & (negative['particle'] == particle)]  
            plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')


    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize = 30)

    plt.xticks(np.arange(0, periods.shape[2], 10))
    plt.yticks(np.arange(0, periods.shape[1], 10))

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.9, f'{int((start_time+frame_intervals*frame_start/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*frame_start/60)%60)//1):0>2}:{int(((frame_intervals*frame_start)%60)//1):0>2}\nFrame: {0}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=40)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(periods[i])
        
        # Remove previous scatter plots
        for collection in fig.gca().collections:
            collection.remove()
    
        if tracking is not False:
            positive, negative = tracking
            colormap_p, _ = create_tracking_colormap(positive)
            colormap_n, _ = create_tracking_colormap(negative)
            for particle in positive[positive['frame'] == i]['particle']:
                particle_data = positive[(positive['frame'] == i) & (positive['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_p[particle], marker = 's')
            for particle in negative[negative['frame'] == i]['particle']:
                particle_data = negative[(negative['frame'] == i) & (negative['particle'] == particle)]
                plt.scatter(particle_data['x'], particle_data['y'], s = 500, color=colormap_n[particle], marker = 'o')
                

        # Update iteration number text
        iteration_text.set_text(f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1):0>2}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}:{int(((frame_intervals*(frame_start+i))%60)//1):0>2}\nFrame: {i}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = periods.shape[0],
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
            im = axs[j].imshow(a, cmap = 'twilight')
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


def animate_correlation(binary: np.ndarray, filename: str, nSecs: float, txy_values, clustering):
    
    iterations = binary.shape[0]
    fps = iterations/nSecs

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(20,17) )
    
    a = binary[0]

    im = plt.imshow(a, vmin = 0, vmax = 1)


        
    # Get unique labels (including -1 for noise)
    unique_labels = np.unique(clustering.labels_)

    # Create a list of colors from the tuple and create a cyclic iterator of colors
    colors = list(plt.cm.tab20.colors)  # Convert tuple to list
    color_cycle = itertools.cycle(colors)

    # Assign a color to each label (cluster) using the cyclic color iterator
    label_color_map = {label: next(color_cycle) for label in unique_labels}
    # Reset color for noise to black, if present in labels
    if -1 in label_color_map:
        label_color_map[-1] = 'k'

    # Filter txy_values and labels for the current time
    current_time_indices = np.where(txy_values[:, 0] == 0)
    current_time_values = txy_values[current_time_indices]
    current_labels = clustering.labels_[current_time_indices]
    
    # Plot each cluster with a different color
    unique_labels = np.unique(current_labels)
    for label in unique_labels:
        # Filter points belonging to the current label
        label_indices = np.where(current_labels == label)
        points = current_time_values[label_indices]
        
        color = label_color_map[label]
        
        # Scatter plot for points
        plt.scatter(points[:, 2], points[:, 1], s=10, color=color, label=f"Cluster {label}" if label != -1 else "Noise")
    
    plt.axis('off')

    # Create text element to display iteration number, centered and with larger font
    iteration_text = plt.text(0.5, 0.9, f'Frame: {0}', transform=plt.gcf().transFigure, horizontalalignment='center', fontsize=40)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(binary[i])
        
        # Remove previous scatter plots
        for collection in fig.gca().collections:
            collection.remove()

            

        # Filter txy_values and labels for the current time
        current_time_indices = np.where(txy_values[:, 0] == i)
        current_time_values = txy_values[current_time_indices]
        current_labels = clustering.labels_[current_time_indices]
        
        # Plot each cluster with a different color
        unique_labels = np.unique(current_labels)
        for label in unique_labels:
            # Filter points belonging to the current label
            label_indices = np.where(current_labels == label)
            points = current_time_values[label_indices]
            
            color = label_color_map[label]
            
            # Scatter plot for points
            plt.scatter(points[:, 2], points[:, 1], s=10, color=color, label=f"Cluster {label}" if label != -1 else "Noise")


        # Update iteration number text
        iteration_text.set_text(f'Frame: {i}')

        return [im, iteration_text]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = binary.shape[0],
                                interval = 1000 / fps, # in ms
                                )

    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])

    print(' Full run done!')
