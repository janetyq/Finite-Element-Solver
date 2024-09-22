import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.colorbar import Colorbar
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from utils.helper import *

class Plotter:
    def __init__(self, nrows=1, ncols=1, figsize=None, title=None):
        if figsize is None:
            size = 5 if max(nrows, ncols) <= 5 else 4
            figsize = (5*ncols, 5*nrows)
        
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=figsize)
        self.fig.suptitle(title)
        if nrows == 1 and ncols == 1:
            self.axs = np.array([self.axs])
        self.axs = self.axs.reshape(nrows, ncols)
        
        self.anims = {} 
        self.cbar_infos = {}

    # function for plotting at a specific index
    def plot(self, mesh, values=None, mode='mesh', idx=(0, 0), title=None, bc=None, clear=False):
        ax = self.axs[idx]
        if clear:
            ax.clear()

        if values is not None:
            values = np.array(values)

        # TODO: check that values/bc are provided for intended mode
        if mode == 'mesh':
            plot_mesh(ax, mesh)
        elif mode == 'boundary':
            plot_boundary(ax, mesh)
        elif mode == 'colored':
            cbar_info = plot_colored(ax, mesh, values, cbar_info=self.cbar_infos.get(idx, None))
            self.cbar_infos[idx] = cbar_info
        elif mode == 'surface':
            ax = change_ax_to_ax3d(ax, self.fig, self.axs.shape, idx)
            self.axs[idx] = ax
            plot_surface(ax, mesh, values)
        elif mode == 'arrows':
            plot_arrows(ax, mesh, values) # inside arrows, assert the correct shape
        elif mode == 'bc':
            plot_bc(ax, mesh, bc)
        else:
            raise ValueError(f'Invalid plot mode: {mode}')

        ax.set_title(title) # overrides any existing title

    def plot_highlights(self, mesh, idxs_list, color_list, label_list, mode='vertices', idx=(0, 0)):
        if not (len(idxs_list) == len(color_list) == len(label_list)):
            raise ValueError("idxs_list, color_list, and label_list must have the same length.")

        ax = self.axs[idx] if isinstance(self.axs, np.ndarray) else self.axs
        plot_highlight(ax, mesh, idxs_list, color_list, label_list, mode=mode)


    # Specialty plotting
    def plot_animation(self, mesh, values, mode='colored', idx=(0, 0), titles=None, cbar_lims=(0, 1)):
        if titles is None:
            titles = [str(i) for i in range(len(values))]

        # sets up colorbar for animation with desired limits
        self.cbar_infos[idx] = setup_colorbar(self.axs[idx], cbar_lims, label=None)
        self.plot(mesh, values[0], mode=mode, idx=idx, title=titles[0])

        def update(frame):
            self.plot(mesh, values[frame], mode=mode, idx=idx, title=titles[frame], clear=True)

        self.anims[idx] = FuncAnimation(self.fig, update, frames=range(len(values)), blit=False, repeat=True)

    def get_ax(self, idx=(0, 0)):
        return self.axs[idx]

    def format_axs(self):
        for ax in self.axs.ravel():
            ax.ticklabel_format(useOffset=False)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if hasattr(ax, 'get_zlim'):
                ax.set_label('z')
                ax.set_aspect('equalxy')
            else:
                ax.set_aspect('equal')

            if any(ax.get_legend_handles_labels()[1]):
                ax.legend()

    def show(self):
        self.format_axs()
        plt.show()

    def save(self, path):
        self.format_axs()
        plt.savefig(path)

        # TODO: animation saving not supported yet

    def close(self):
        plt.close()
        