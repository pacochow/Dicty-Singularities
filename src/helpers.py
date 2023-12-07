import matplotlib.pyplot as plt

def create_colormap():
    cmap = plt.cm.hsv.copy()
    cmap.set_bad(color = 'black')

    return cmap