import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from utils.linalg import *
from utils.helper import *


class Mesh:
    def __init__(self, points, faces, boundary):
        self.points = np.array(points)
        self.faces = np.array(faces)
        self.boundary = np.array(boundary)
        self.total_area = self.calculate_total_area()

    @classmethod
    def load(cls, path='test_mesh.pkl'):
        with open(path, 'rb') as f:
            return cls(*pickle.load(f))

    def save(self, path='test_mesh.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_info(), f)
        print(f'Saved mesh to {path}')

    def get_info(self):
        return self.points, self.faces, self.boundary

    def __repr__(self):
        return f'Mesh(points={self.points}, faces={self.faces}, boundary={self.boundary})'

    def copy(self):
        return Mesh(self.points.copy(), self.faces.copy(), self.boundary.copy())


    # PLOTTING
    def plot(self, title=None, ax=None, show=True, color='black', linewidth=1):
        if ax is None:
            fig, ax = plt.subplots()

        # triangulation and boundary
        ax.triplot(self.points[:, 0], self.points[:, 1], self.faces, color=color, linewidth=linewidth)
        for seg in self.boundary:
            ax.plot(self.points[seg, 0], self.points[seg, 1], color=color, linewidth=1.5*linewidth)
            # for testing normals
            # vec = self.points[seg[1]] - self.points[seg[0]]
            # point = np.mean(self.points[seg], axis=0) + 0.5 * np.array([-vec[1], vec[0]])            
            # ax.scatter(point[0], point[1], color='green', s=5)

        # ax settings
        ax.set_title(title)
        ax.set_aspect('equal')

        if show:
            plt.show()
        return ax

    def plot_colored(self, u, title=None, contour=0, colorscale=None, ax=None, show=True, cbar_label=''):
        '''Plots 2d triangulated mesh with colored faces/vertices according to u'''
        if ax is None:
            fig, ax = plt.subplots()

        # triangulation and color plot
        plot_triangulation = Triangulation(self.points[:, 0], self.points[:, 1], triangles=self.faces)
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis')
        cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
        cbar.set_label('', rotation=270)
        if contour > 0:
            contour_levels = np.linspace(min(u), max(u), contour)  # Adjust the number of levels as needed
            ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)

        if show:
            plt.show()
        
        return ax
    
    def plot_surface(self, u, title=None, ax=None, show=True):
        '''Plots 2d surface mesh in 3d with colored faces according to u'''
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            # set ax projection as 3d
            # TODO: implement this
            pass
        plot_triangulation = Triangulation(self.points[:, 0], self.points[:, 1], triangles=self.faces)
        ax.plot_trisurf(plot_triangulation, u, cmap='viridis')

        # ax settings
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_xlim([min(self.points[:, 0]), max(self.points[:, 0])])
        ax.set_ylim([min(self.points[:, 1]), max(self.points[:, 1])])
        
        if show:
            plt.show()

    def plot_arrows(self, u, title=None, ax=None, show=True, color='red'):
        if ax is None:
            fig, ax = plt.subplots()
        face_points = np.mean(self.points[self.faces], axis=1)
        ax.quiver(face_points[:, 0], face_points[:, 1], u[:, 0], u[:, 1], color=color, alpha=0.7)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)
        
        if show:
            plt.show()
        return ax

    def plot_colored_animation(self, t_values, u_values, title=None, contour=0, cbar_label=None, fixed_cbar=False):
        fig, ax = plt.subplots()
        
        # ax settings
        ax.set_aspect('equal')
        ax.ticklabel_format(useOffset=False)

        # plot first frame
        ax.set_title(f'{title} t = 0')
        plot_triangulation = Triangulation(self.points[:, 0], self.points[:, 1], triangles=self.faces)
        tripcolor = ax.tripcolor(plot_triangulation, u_values[0], cmap='viridis')
        cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
        cbar.set_label(cbar_label, rotation=270)

        u_init_min, u_init_max = min(u_values[0]), max(u_values[0])

        def update(frame):
            ax.clear()
            ax.set_title(f'{title} t = {t_values[frame]:.3f}')
            u = u_values[frame]    
            if not fixed_cbar:
                tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis')
                cbar.mappable.set_clim(vmin=u.min(), vmax=u.max())
            else:
                tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis', vmin=u_init_min, vmax=u_init_max)
            if contour > 0:
                contour_levels = np.linspace(min(u), max(u), contour)
                ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')
            return ax

        ani = FuncAnimation(fig, update, frames=range(len(u_values)), blit=False, repeat=True)

        plt.show()

    def plot_surface_animation(self, t_values, u_values, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax settings
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_aspect('equal')

        plot_triangulation = Triangulation(self.points[:, 0], self.points[:, 1], triangles=self.faces)

        def update(frame):
            ax.clear()
            ax.set_title(f'{title} t = {t_values[frame]:.3f}')
            surf = ax.plot_trisurf(plot_triangulation, u_values[frame], cmap='viridis')
            return surf

        ani = FuncAnimation(fig, update, frames=range(len(t_values)), blit=False, repeat=True, interval=400)
        plt.show()


    # METRICS
    def calculate_total_value(self, u):
        total_value = 0
        for face in self.faces:
            area = calculate_triangle_area(self.points[face])
            u_value = np.mean(u[face])
            total_value += area * u_value
        return total_value

    def calculate_total_area(self):
        total_area = 0
        for face in self.faces:
            area = calculate_triangle_area(self.points[face])
            total_area += area
        return total_area

    def calculate_mean_value(self, u):
        return self.calculate_total_value(u) / self.total_area

    # TODO: check if this works, from copilot
    # def calculate_gradient(self, u):
    #     gradient = np.zeros((len(self.faces), 2))
    #     for face_idx, face in enumerate(self.faces):
    #         element = self.points[face]
    #         area = calculate_triangle_area(element)
    #         for i in range(3):
    #             u_value = u[face[i]]
    #             edge10 = element[i] - element[(i+1)%3]
    #             gradient[face_idx] += u_value * edge10
    #         gradient[face_idx] /= (2 * area)
    #     return gradient

    def calculate_gradient(self, u):
        gradient = np.zeros((len(self.faces), 2))
        for face_idx, face in enumerate(self.faces):
            element = self.points[face]
            area = calculate_triangle_area(element)
            for i in range(3):
                u_value = u[face[i]]
                edge10 = element[i] - element[(i+1)%3]
                edge12 = element[(i+2)%3] - element[(i+1)%3]
                cross = calc_cross(edge12, edge10)
                edge_center = (element[(i+1) % 3] + element[(i+2) % 3]) / 2

                # TODO: sign flipped?
                if cross > 0:
                    gradient[face_idx] += -np.array([-edge12[1], edge12[0]]) * u_value / (2*area)
                else:
                    gradient[face_idx] += -np.array([edge12[1], -edge12[0]]) * u_value / (2*area)    
        return gradient

    def calculate_dirichlet_energy(self, u):
        energy = 0
        u_gradient = self.calculate_gradient(u)
        for face_idx, face in enumerate(self.faces):
            area = calculate_triangle_area(self.points[face])
            energy += 1/2 * area * calc_dot(u_gradient[face_idx], u_gradient[face_idx])
        return energy

    def calculate_energy(self, u, dudt):
        # TODO: not conserved in wave eqn for some reason
        dirichlet_energy = self.calculate_dirichlet_energy(u)
        kinetic_energy = 1/2 * self.calculate_total_value(dudt**2)
        return dirichlet_energy + kinetic_energy
