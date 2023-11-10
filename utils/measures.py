import numpy as np
import matplotlib.pyplot as plt
from utils.helper import *
from utils.linalg import *

## MEASURES ##
# TODO
# mean value, dirichlet energy
# verify gradient works


def calculate_total_value(points, faces, u):
    '''
    Takes in a mesh with u defined on the NODES
    Returns the total value of u on the mesh weighted by area
    '''
    N = len(points)
    total_value = 0
    for face in faces:
        element = points[face]
        area = calculate_triangle_area(element)
        u_value = np.mean(u[face])
        total_value += area * u_value
    return total_value

def calculate_total_area(points, faces):
    '''
    Takes in a mesh
    Returns the total area of the mesh
    '''
    N = len(points)
    total_area = 0
    for face in faces:
        element = points[face]
        area = calculate_triangle_area(element)
        total_area += area
    return total_area

def calculate_mean_value(points, faces, u):
    '''
    Takes in a mesh with u defined on the NODES
    Returns the mean value of u on the mesh weighted by area
    '''
    total_value = calculate_total_value(points, faces, u) 
    total_area = calculate_total_area(points, faces)
    return total_value / total_area # not as efficient, but nicer


def calculate_gradient(points, faces, u):
    '''
    Takes in a mesh with u defined on the NODES
    Returns the gradient of u on the FACES
    '''
    gradient = np.zeros((len(faces), 2))
    for face_idx, face in enumerate(faces):
        element = points[face]
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

def calculate_dirichlet_energy(points, faces, u):
    '''
    Takes in a mesh with u defined on the NODES
    Returns the dirichlet energy of u on the mesh

    Dirichlet energy is the integral of 1/2 * the squared gradient of u
    '''
    energy = 0
    u_gradient = calculate_gradient(points, faces, u)
    for face_idx, face in enumerate(faces):
        area = calculate_triangle_area(points[face])
        energy += 1/2 * area * calc_dot(u_gradient[face_idx], u_gradient[face_idx])
    return energy

def calculate_energy(points, faces, u, dudt):
    '''
    Takes in a mesh with u, dudt defined on the NODES
    Returns the total energy of u on the mesh
    '''
    dirichlet_energy = calculate_dirichlet_energy(points, faces, u)
    kinetic_energy = 1/2 * calculate_total_value(points, faces, dudt**2)
    return dirichlet_energy + kinetic_energy

