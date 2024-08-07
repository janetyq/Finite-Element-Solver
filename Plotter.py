import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.tri import Triangulation
from matplotlib.colorbar import Colorbar
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Plotter class
# - takes in a mesh and options and plots the mesh or values on the mesh
# - all plotting capabilities are in the plotter class,
#   but some other classes have their own derived plot functions for convenience

class Plotter:
    def __init__(self, mesh, fig=None, ax=None, options=None):
        self.mesh = mesh
        if fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = fig, ax
        self.options = options or {}

        self.cbar = None # used by colored plots

    # Main plotting
    def plot_mesh(self, mode='wireframe', color_elements=None, color_vertices=None):
        if color_elements is not None:      
            for color, e_idxs, label in color_elements:
                first = True
                for e_idx in e_idxs:
                    vertices = self.mesh.vertices[self.mesh.elements[e_idx]]
                    self.ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.2, label=label if first else None)
                    first = False
        if color_vertices is not None:
            for color, vert_idxs, label in color_vertices:
                self.ax.scatter(self.mesh.vertices[vert_idxs, 0], self.mesh.vertices[vert_idxs, 1], color=color, s=5, label=label)
        
        if mode == 'wireframe':
            self._plot_wireframe()
        elif mode == 'solid':
            self._plot_solid()
        self._apply_options()

    def plot_values(self, values, mode='colored'):
        if mode == 'colored' or mode is None:
            self._plot_colored(values)
        elif mode == 'surface':
            self._plot_surface(values)
        elif mode == 'arrows':
            self._plot_arrows(values)
        else:
            raise ValueError(f'Invalid plot_values mode: {mode}')
        self._apply_options()

    # Specialty plotting
    def plot_bc(self, bc, values=None):
        for v_idx, value in bc.dirichlet.items():
            self.ax.plot(self.mesh.vertices[v_idx][0], self.mesh.vertices[v_idx][1], 'ro')
        for v_idx, value in bc.neumann.items():
            self.ax.quiver(self.mesh.vertices[v_idx][0], self.mesh.vertices[v_idx][1], value[0], value[1])
        if values is None:
            self.plot_mesh(mode='wireframe')
        else:
            self.plot_values(values, title='BC')
        self._apply_options()

    def plot_animation(self, values, t_values=None, meshes=None, mode='colored'):
        if not plt.fignum_exists(self.fig.number):
            self.reset()
        save = self.options.get('save', None)
        self.options['title'] =  t_values[0] if t_values is not None else 0
        self.options['show'] = False
        self.options['save'] = False

        if meshes is not None:
            self.mesh = meshes[0]
        self.plot_values(values[0], mode=mode)

        def update(frame):
            self.ax.clear()
            self.options['title'] = t_values[frame] if t_values is not None else frame
            if meshes is not None:
                self.mesh = meshes[frame]
            self.plot_values(values[frame], mode=mode)

        ani = FuncAnimation(self.fig, update, frames=range(len(values)), blit=False, repeat=True)

        if save:
            writervideo = animation.FFMpegWriter(fps=5) 
            ani.save(save, writer=writervideo) 
            plt.close()
        else:
            plt.show()
    
    # Mesh plotting helpers
    def _plot_wireframe(self):
        self.ax.triplot(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements, color='black', linewidth=0.2)
        for seg in self.mesh.boundary:
            self.ax.plot(self.mesh.vertices[seg, 0], self.mesh.vertices[seg, 1], color='black', linewidth=0.5)

    def _plot_solid(self):
        for element in self.mesh.elements:
            vertices = self.mesh.vertices[element]
            center = np.mean(vertices, axis=0)
            vertices = center + 0.9 * (vertices - center)
            # self.ax.fill(vertices[:, 0], vertices[:, 1], 'b-', alpha=0.2)
            vertices = np.append(vertices, [vertices[0]], axis=0)
            self.ax.plot(vertices[:, 0], vertices[:, 1], 'b-', alpha=0.5)
        for edge in self.mesh.boundary:
            self.ax.plot(self.mesh.vertices[[edge[0], edge[1]], 0], self.mesh.vertices[[edge[0], edge[1]], 1], 'k-')

    # Value plotting helpers
    def _plot_colored(self, values, contour=0):
        triangulation = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles=self.mesh.elements)
        tripcolor = self.ax.tripcolor(triangulation, values, cmap='viridis')

        clim = self.options.get('cbar_lim', (min(values), max(values)))
        tripcolor.set_clim(*clim)

        if self.cbar is None:
            self.cbar = plt.colorbar(tripcolor, shrink=0.6)
            self.cbar.set_label(self.options.get('cbar_label', 'value'), rotation=270, labelpad=15)

        if contour > 0:
            self.ax.tricontour(triangulation, values, levels=np.linspace(min(values), max(values), contour), colors='k', linestyles='solid')

    def _plot_surface(self, values):
        if not hasattr(self.ax, 'plot_trisurf'):
            self.ax.remove()
            self.ax = self.fig.add_subplot(111, projection='3d')
        values = np.array(values)
        assert values.shape == (len(self.mesh.vertices),), f'Invalid values shape: {values.shape}'
        triangulation = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles=self.mesh.elements)
        surf = self.ax.plot_trisurf(triangulation, values, cmap='viridis')

    def _plot_arrows(self, values):
        element_vertices = np.mean(self.mesh.vertices[self.mesh.elements], axis=1)
        self.ax.quiver(element_vertices[:, 0], element_vertices[:, 1], values[:, 0], values[:, 1], alpha=0.5, scale=10)

    def _apply_options(self):
        self.ax.set_title(self.options.get('title', None))
        self.ax.ticklabel_format(useOffset=False)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        if hasattr(self.ax, 'get_zlim'): # TODO: doesn't seem equal to me
            self.ax.set_label('z')
            self.ax.set_aspect('equalxy')
        else:
            self.ax.set_aspect('equal')

        # check if legend is needed
        if any(self.ax.get_legend_handles_labels()[1]):
            self.ax.legend()

        # Save or show the plot
        save = self.options.get('save', None)
        if save:
            plt.savefig(save)
            plt.close()
        elif self.options.get('show', True):
            plt.show()
        
    def reset(self):
        self.fig, self.ax = plt.subplots()
        self.cbar = None
