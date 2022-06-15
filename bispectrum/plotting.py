from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


def image_grid(data, shape=(10,10), figsize=(10,10), cmap='Greys_r', share_range=True, interpolation=None, save_name=None):

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=shape,  # creates 10x10 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    if share_range:
        vmin = data.min()
        vmax = data.max()
        
    for ax, im in zip(grid, data):
        # Iterating over the grid returns the Axes.
        if share_range:
            ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
        else:
            ax.imshow(im, cmap=cmap, interpolation=interpolation)
        ax.set_axis_off()
        
    if save_name is not None:
        plt.savefig(save_name)
        