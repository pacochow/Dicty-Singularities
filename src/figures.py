from src.helpers import *
import matplotlib.pyplot as plt


def create_stills(frames: np.ndarray, filename: str, frame_numbers: list, format: tuple, dims: tuple = (15, 7), tracking = False):
    

    start_time = 3
    frame_intervals = 15
    frame_start = frame_numbers[0]
    times = [f'{int((start_time+frame_intervals*(frame_start+i)/3600)//1)}:{int(((60*(start_time-(start_time//1))+frame_intervals*(frame_start+i)/60)%60)//1):0>2}:{int(((frame_intervals*(frame_start+i))%60)//1):0>2}' 
             for i in frame_numbers]
    
    if format[0] == 1:
        # Plotting
        fig, axes = plt.subplots(1, format[1], figsize=(15, 5)) 

        for i, ax in enumerate(axes):
            ax.imshow(frames[frame_numbers[i]], cmap = 'twilight', vmin = -np.pi, vmax = np.pi)
            if frame_numbers[i]==899:
                axes[i//format[1], i%format[1]].set_title("06:45:00", fontsize = 22)
            else:
                ax.set_title(f"{times[i]}")
            ax.axis('off')  # To turn off axis numbers
            if tracking is not False:
                        positive, negative = tracking
                        colormap_p, _ = create_tracking_colormap(positive)
                        colormap_n, _ = create_tracking_colormap(negative)

                        for particle in positive[positive['frame'] == frame_numbers[i]]['particle']:
                            particle_data = positive[(positive['frame'] == frame_numbers[i]) & (positive['particle'] == particle)]
                            ax.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_p[particle], marker = 's')
                        for particle in negative[negative['frame'] == frame_numbers[i]]['particle']:
                            particle_data = negative[(negative['frame'] == frame_numbers[i]) & (negative['particle'] == particle)]  
                            ax.scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_n[particle], marker = 'o')
    else:
        fig, axes = plt.subplots(format[0], format[1], figsize = dims)
        
        for i in range(format[0]*format[1]):
            axes[i//format[1], i%format[1]].axis('off')  # To turn off axis numbers
            if i > len(frame_numbers)-1:
                break
            axes[i//format[1], i%format[1]].imshow(frames[frame_numbers[i]], cmap = 'twilight', vmin =-np.pi, vmax = np.pi)
            if tracking is not False:
                        positive, negative = tracking
                        colormap_p, _ = create_tracking_colormap(positive)
                        colormap_n, _ = create_tracking_colormap(negative)

                        for particle in positive[positive['frame'] == frame_numbers[i]]['particle']:
                            particle_data = positive[(positive['frame'] == frame_numbers[i]) & (positive['particle'] == particle)]
                            axes[i//format[1], i%format[1]].scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_p[particle], marker = 's')
                        for particle in negative[negative['frame'] == frame_numbers[i]]['particle']:
                            particle_data = negative[(negative['frame'] == frame_numbers[i]) & (negative['particle'] == particle)]  
                            axes[i//format[1], i%format[1]].scatter(particle_data['x'], particle_data['y'], s = 30, color=colormap_n[particle], marker = 'o')

            if frame_numbers[i]==frames.shape[0]-1:
                axes[i//format[1], i%format[1]].set_title("06:45:00", fontsize = 22)
            else:
                axes[i//format[1], i%format[1]].set_title(f"{times[i]}", fontsize = 22)
            axes[i//format[1], i%format[1]].set_xticks([])
            axes[i//format[1], i%format[1]].set_yticks([])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()