"""Numerical utilities: source/field functions, the SIMP smoothing matrix,
finite-difference gradient/Hessian checks, and small dev helpers (timer, color).
"""
import logging
from math import cos, pi

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def bump_function(vertices, center, mag=100, size=0.5):
    return np.array([mag*cos(pi/2*np.linalg.norm(point - center)/size) if np.linalg.norm(point - center) < size else 0 for point in vertices])


def calculate_smoothing_matrix(mesh, r):
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    weight_matrix = np.maximum(0, r - distances)
    normalized_weight_matrix = weight_matrix / (weight_matrix.sum(axis=1)[:, np.newaxis] + 1e-16)
    return normalized_weight_matrix


# Gradient checking - TODO: make faster
def check_gradient(function, gradient, input_shape):
    u = np.random.random(input_shape)
    computed_gradient = gradient(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_gradient = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = function(u + eps * direction)
            eval_m = function(u - eps * direction)
            numerical_gradient.append((eval_p - eval_m) / (2 * eps))
        numerical_gradient = np.array(numerical_gradient).reshape(computed_gradient.shape)
        # print(f'numerical_gradient: {numerical_gradient} \ncomputed_gradient: {computed_gradient}')
        errors_list.append(np.linalg.norm(numerical_gradient - computed_gradient))

    plt.title('Gradient check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()


def check_hessian(gradient, hessian, input_shape):
    u = np.random.random(input_shape)
    computed_hessian = hessian(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_hessian = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = gradient(u + eps * direction)
            eval_m = gradient(u - eps * direction)
            numerical_hessian.append((eval_p - eval_m) / (2 * eps))
        numerical_hessian = np.array(numerical_hessian).reshape(computed_hessian.shape)
        errors_list.append(np.linalg.norm(numerical_hessian - computed_hessian))

    plt.title('Hessian check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()


# Decorators
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info('%s took %s seconds', func.__name__, end - start)
        return result
    return wrapper


# ANSI terminal colors for pretty-printing
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
