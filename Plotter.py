import numpy as np
import matplotlib.pyplot as plt

from Solution import *
from utils.helper import *

class Plotter:
    def __init__(self, solution):
        self.solution = solution

    @plot_decorator
    def plot_colored(self, name, idx=None, fig=None, ax=None, cbar=None, cbar_lim=None, cbar_label=None, contour=0, deformed=False, *args, **kwargs):
        mesh = self.solution.get_values('deformed_mesh', idx) if deformed else self.solution.mesh
        values = self.solution.get_values(name, idx, mode='vertex')

        plot_triangulation = Triangulation(mesh.points[:, 0], mesh.points[:, 1], triangles=mesh.faces)
        tripcolor = ax.tripcolor(plot_triangulation, values, cmap='viridis')

        cbar = cbar or plt.colorbar(tripcolor, ax=ax, shrink=0.6)
        cbar.set_label(cbar_label or name, rotation=270, labelpad=15)
        clim = cbar_lim or (values.min(), values.max())
        cbar.mappable.set_clim(*clim)

        if contour > 0:
            ax.tricontour(plot_triangulation, values, levels=np.linspace(min(values), max(values), contour), colors='k', linestyles='solid')
        
        return fig, ax, cbar

    @plot_decorator
    def plot_surface(self, name, idx=None, fig=None, ax=None, deformed=False, *args, **kwargs):
        ax = fig.add_subplot(111, projection='3d') # TODO: this is funky

        mesh = self.get_values('deformed_mesh', idx) if deformed else self.mesh
        values = self.get_values(name, idx)

        plot_triangulation = Triangulation(mesh.points[:, 0], mesh.points[:, 1], triangles=mesh.faces)
        surf = ax.plot_trisurf(plot_triangulation, values, cmap='viridis')

        ax.set_zlabel(name)
        ax.set_xlim([min(mesh.points[:, 0]), max(mesh.points[:, 0])])
        ax.set_ylim([min(mesh.points[:, 1]), max(mesh.points[:, 1])])
        
        return fig, ax, None

    @plot_decorator
    def plot_arrows(self, name, idx=None, color='blue', fig=None, ax=None, *args, **kwargs):
        values = self.get_values(name, idx)

        face_points = np.mean(self.mesh.points[self.mesh.faces], axis=1)
        ax.quiver(face_points[:, 0], face_points[:, 1], values[:, 0], values[:, 1], color=color, alpha=0.5)
        
        return fig, ax

    def plot_animation(self, name, title='Animation', cbar_label='Value', fixed_cbar=False, cbar_lim=None, mode='color', fps=15, deformed=False, save=None):
        t_values = self.get_values('t_values') if 't_values' in self.solution.values else np.arange(len(self.get_values(name)))
        u_values = self.get_values(name)
        
        if mode == 'surface':
            fig, ax, _ = self.plot_surface(name, idx=0, projection='3d', title=f'{title} t = {t_values[0]:.3f}', deformed=deformed, show=False)
            cbar = None

            def update(frame, fig, ax, cbar):
                ax.clear()
                fig, ax, _ = self.plot_surface(name, idx=frame, title=f'{title} t = {t_values[frame]:.3f}', fig=fig, ax=ax, deformed=deformed, show=False)
                return ax
        else:
            fig, ax, cbar = self.plot_colored(name, idx=0, title=f'{title} t = 0', deformed=deformed, show=False)
            if cbar_lim is None:
                cbar_lim = (u_values[0].min(), u_values[1].max()) if not fixed_cbar else None

            def update(frame, fig, ax, cbar):
                ax.clear()
                fig, ax, cbar = self.plot_colored(name, idx=frame, title=f'{title} t = {frame}', cbar=cbar, cbar_lim=cbar_lim, fig=fig, ax=ax, deformed=deformed, show=False)
                return ax

        ani = FuncAnimation(fig, update, frames=range(len(t_values)), fargs=(fig, ax, cbar), blit=False, repeat=True)

        if save:
            writervideo = animation.FFMpegWriter(fps=5) 
            ani.save(save, writer=writervideo) 
            plt.close()
        else:
            plt.show()
