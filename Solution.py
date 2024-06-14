from datetime import datetime
import matplotlib.animation as animation 
import numpy as np
import pickle

from Mesh import *

class Solution:
    def __init__(self, mesh, values=None):
        self.mesh = mesh
        self.values = values if values is not None else {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return cls(pickle.load(f))

    def __reduce__(self):
        return (self.__class__, (self.mesh, self.values))

    def set_values(self, name, value):
        self.values[name] = value

    def reset(self):
        self.values = {}

    def calc_gradient(self, name):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')
        u = self.values[name]
        gradient = np.zeros((len(self.mesh.faces), 2))
        for face_idx, face in enumerate(self.mesh.faces):
            element = self.mesh.points[face]
            for i in range(3):
                edge_i = element[(i+1)%3] - element[i]
                edge_j = element[(i+2)%3] - element[i]
                sign = calc_cross(edge_i, edge_j) / np.abs(calc_cross(edge_i, edge_j))
                gradient[face_idx] += sign * u[face[(i+2)%3]] * np.array([-edge_i[1], edge_i[0]]) / (2*self.mesh.areas[face_idx])
        self.values["grad_" + name] = gradient
        return gradient

    def _convert_vertex_values_to_face_values(self, vertex_values):
        assert len(vertex_values) == len(self.mesh.points)
        face_values = np.zeros(len(self.mesh.faces))
        for face_idx, face in enumerate(self.mesh.faces):
            face_values[face_idx] = np.mean([vertex_values[v_idx] for v_idx in face])
        return face_values

    def _convert_face_values_to_vertex_values(self, face_values):
        assert len(face_values) == len(self.mesh.faces)
        vertex_values = np.zeros(len(self.mesh.points))
        for face_idx, face in enumerate(self.mesh.faces):
            for v_idx in face:
                vertex_values[v_idx] = face_values[face_idx]
        return vertex_values

    def get_values(self, name):
        if name not in self.values:
            print('solution contains:', self.values.keys())
            raise ValueError(f'{name} not found in solution')
        return self.values[name]

    def plot(self, name, title='Solution', ax=None, show=True, contour=20):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')
        self.plot_colored(name, title=title, ax=ax, show=show, contour=contour)
    
    def plot_colored(self, name, idx=None, title=None, contour=0, ax=None, show=True, cbar=None, cbar_lim=None, cbar_label=None, save=None, mesh=None):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')
        if mesh is None:
            mesh = self.mesh

        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.clear()

        u_values = self.values[name]
        if idx is not None:
            u_values = u_values[idx]

        plot_triangulation = Triangulation(mesh.points[:, 0], mesh.points[:, 1], triangles=mesh.faces)
        tripcolor = ax.tripcolor(plot_triangulation, u_values, cmap='viridis')
        if cbar is None:
            cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
            cbar.set_label(name if cbar_label is None else cbar_label, rotation=270, labelpad=15)
        if cbar_lim is not None:
            cbar.mappable.set_clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        else:
            cbar.mappable.set_clim(vmin=u_values.min(), vmax=u_values.max())
        if contour > 0:
            if len(u_values) == len(self.mesh.faces):
                u_values = self._convert_face_values_to_vertex_values(u_values)
            ax.tricontour(plot_triangulation, u_values, levels=np.linspace(min(u_values), max(u_values), contour), colors='k', linestyles='solid')

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)
        
        if save:
            plt.savefig(save)
            plt.close()
        elif show:
            plt.show()
        
        return ax, plot_triangulation, cbar

    def plot_surface(self, name, idx=None, title=None, ax=None, show=True, save=None):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}) if ax is None else (ax.figure, ax)
        ax.clear()

        u_values = self.values[name]
        if idx is not None:
            u_values = u_values[idx]
        
        plot_triangulation = Triangulation(self.mesh.points[:, 0], self.mesh.points[:, 1], triangles=self.mesh.faces)
        surf = ax.plot_trisurf(plot_triangulation, u_values, cmap='viridis')

        # ax settings
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(name)
        ax.set_xlim([min(self.mesh.points[:, 0]), max(self.mesh.points[:, 0])])
        ax.set_ylim([min(self.mesh.points[:, 1]), max(self.mesh.points[:, 1])])
        
        if save:
            plt.savefig(save)
            plt.close()
        elif show:
            plt.show()
        return ax, surf

    def plot_arrows(self, name, idx=None, title=None, ax=None, show=True, color='blue'):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')

        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.clear()

        u_values = self.values[name]
        if idx is not None:
            u_values = u_values[idx]

        face_points = np.mean(self.mesh.points[self.mesh.faces], axis=1)
        ax.quiver(face_points[:, 0], face_points[:, 1], u_values[:, 0], u_values[:, 1], color=color, alpha=0.5)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)
        
        if show:
            plt.show()
        return ax

    def plot_animation(self, name, title='Animation', cbar_label='Value', cbar_lim=None, surface=False, savefile=None, fps=15, save=None, show=True):
        if name not in self.values:
            raise ValueError(f'{name} not found in solution')

        u_values = self.values[name].copy()
        t_values = np.arange(len(u_values))

        if len(u_values[0]) == len(self.mesh.faces):
            for i, u in enumerate(u_values):
                u_values[i] = self._convert_face_values_to_vertex_values(u)

        if surface:
            ax, surf = self.plot_surface(name, idx=0, title=f'{title} t = 0', show=False)
            fig = ax.figure
            cbar = None

            def update(frame, ax, cbar):
                ax, surf = self.plot_surface(name, idx=frame, title=f'{title} t = {t_values[frame]:.3f}', ax=ax, show=False)
                return surf
        else:
            ax, plot_triangulation, cbar = self.plot_colored(name, idx=0, title=f'{title} t = 0', show=False)
            fig = ax.figure
            cbar_lim = (min(u_values[0]), max(u_values[0])) if cbar_lim is None else cbar_lim

            def update(frame, ax, cbar):
                ax, plot_triangulation, cbar = self.plot_colored(name, idx=frame, title=f'{title} t = {frame}', cbar=cbar, cbar_lim=cbar_lim, ax=ax, show=False)
                return ax

        ani = FuncAnimation(fig, update, frames=range(len(u_values)), fargs=(ax, cbar), blit=False, repeat=True)

        if savefile is not None:
            writervideo = animation.FFMpegWriter(fps=fps) 
            ani.save(savefile, writer=writervideo)

        
        if save:
            writervideo = animation.FFMpegWriter(fps=5) 
            ani.save(save, writer=writervideo) 
            plt.close()
        elif show:
            plt.show()

    def plot_deformed(self, name, title='Deformed', ax=None, show=True, save=None):
        if 'deformed_mesh' not in self.values:
            raise ValueError('Deformed mesh not found in solution')
        self.plot_colored(name, title=title, ax=ax, show=show, mesh=self.values['deformed_mesh'], save=save)
        