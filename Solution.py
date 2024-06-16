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
    
    def __copy__(self):
        values_copy = {k: v.copy() for k, v in self.values.items()}
        return self.__class__(self.mesh.copy(), values_copy)

    def __reduce__(self):
        return (self.__class__, (self.mesh, self.values))

    def get_values(self, name, idx=None, mode=None):
        if name is None:
            return np.zeros(len(self.mesh.faces))
        elif name not in self.values:
            print('--> contains:', self.values.keys())
            raise ValueError(f'{name} not found in solution')
        
        values = self.values[name][idx] if idx is not None else self.values[name]
        if mode is None:
            return values
        elif mode == 'face':
            if len(values) == len(self.mesh.faces):
                return values
            elif len(values) == len(self.mesh.points):
                return self._convert_vertex_values_to_face_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')
        elif mode == 'vertex':
            if len(values) == len(self.mesh.points):
                return values
            elif len(values) == len(self.mesh.faces):
                return self._convert_face_values_to_vertex_values(values)
            else:
                raise ValueError(f'Invalid values shape for mode {mode}')


    def set_values(self, name, value):
        self.values[name] = value

    def reset(self):
        self.values = {}

    def calc_gradient(self, name):
        values = self.get_values(name)
        gradient = np.zeros((len(self.mesh.faces), 2))
        for face_idx, face in enumerate(self.mesh.faces):
            element = self.mesh.points[face]
            for i in range(3):
                edge_i = element[(i+1)%3] - element[i]
                edge_j = element[(i+2)%3] - element[i]
                sign = calc_cross(edge_i, edge_j) / np.abs(calc_cross(edge_i, edge_j))
                gradient[face_idx] += sign * values[face[(i+2)%3]] * np.array([-edge_i[1], edge_i[0]]) / (2*self.mesh.areas[face_idx])
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

    # def plot(self, name, title='Solution', ax=None, show=True, contour=20):
    #     if name not in self.values:
    #         raise ValueError(f'{name} not found in solution')
    #     self.plot_colored(name, title=title, ax=ax, show=show, contour=contour)
    
    @plot_decorator
    def plot_colored(self, name, idx=None, title=None, fig=None, ax=None, cbar=None, cbar_lim=None, cbar_label=None, contour=0, mesh=None, *args, **kwargs):
        values = self.get_values(name, idx)
        mesh = self.mesh if mesh is None else mesh

        plot_triangulation = Triangulation(mesh.points[:, 0], mesh.points[:, 1], triangles=mesh.faces)
        tripcolor = ax.tripcolor(plot_triangulation, values, cmap='viridis')
        if cbar is None:
            cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
            cbar.set_label(name if cbar_label is None else cbar_label, rotation=270, labelpad=15)
        if cbar_lim is not None:
            cbar.mappable.set_clim(vmin=cbar_lim[0], vmax=cbar_lim[1])
        else:
            cbar.mappable.set_clim(vmin=values.min(), vmax=values.max())
        if contour > 0:
            if len(values) == len(self.mesh.faces):
                values = self._convert_face_values_to_vertex_values(values)
            ax.tricontour(plot_triangulation, values, levels=np.linspace(min(values), max(values), contour), colors='k', linestyles='solid')
        
        return fig, ax, cbar

    @plot_decorator
    def plot_surface(self, name, idx=None, fig=None, ax=None, *args, **kwargs):
        values = self.get_values(name, idx)

        plot_triangulation = Triangulation(self.mesh.points[:, 0], self.mesh.points[:, 1], triangles=self.mesh.faces)
        surf = ax.plot_trisurf(plot_triangulation, values, cmap='viridis')

        ax.set_zlabel(name)
        ax.set_xlim([min(self.mesh.points[:, 0]), max(self.mesh.points[:, 0])])
        ax.set_ylim([min(self.mesh.points[:, 1]), max(self.mesh.points[:, 1])])
        
        return fig, ax, None

    @plot_decorator
    def plot_arrows(self, name, idx=None, color='blue', fig=None, ax=None, *args, **kwargs):
        values = self.get_values(name, idx)

        face_points = np.mean(self.mesh.points[self.mesh.faces], axis=1)
        ax.quiver(face_points[:, 0], face_points[:, 1], u_values[:, 0], u_values[:, 1], color=color, alpha=0.5)
        
        return fig, ax

    def plot_animation(self, name, title='Animation', cbar_label='Value', fixed_cbar=False, cbar_lim=None, mode='color', fps=15, save=None):
        if "t_values" in self.values:
            t_values = self.get_values('t_values')
        else:
            t_values = np.arange(len(self.get_values(name)))
        u_values = self.get_values(name)
        
        if mode == 'surface':
            fig, ax, _ = self.plot_surface(name, idx=0, projection='3d', title=f'{title} t = {t_values[0]:.3f}', show=False)
            cbar = None

            def update(frame, fig, ax, cbar):
                ax.clear()
                fig, ax, _ = self.plot_surface(name, idx=frame, title=f'{title} t = {t_values[frame]:.3f}', fig=fig, ax=ax, show=False)
                return ax
        else:
            fig, ax, cbar = self.plot_colored(name, idx=0, title=f'{title} t = 0', show=False)
            if cbar_lim is None:
                cbar_lim = (u_values[0].min(), u_values[1].max()) if not fixed_cbar else None

            def update(frame, fig, ax, cbar):
                fig, ax, cbar = self.plot_colored(name, idx=frame, title=f'{title} t = {frame}', cbar=cbar, cbar_lim=cbar_lim, fig=fig, ax=ax, show=False)
                return ax

        ani = FuncAnimation(fig, update, frames=range(len(t_values)), fargs=(fig, ax, cbar), blit=False, repeat=True)

        if save:
            writervideo = animation.FFMpegWriter(fps=5) 
            ani.save(save, writer=writervideo) 
            plt.close()
        else:
            plt.show()

    @plot_decorator
    def plot_deformed(self, name, idx=None, fig=None, ax=None, *args, **kwargs):
        deformed_mesh = self.get_values('deformed_mesh', idx)
        self.plot_colored(name, idx=idx, mesh=deformed_mesh, fig=fig, ax=ax, *args, **kwargs)
        