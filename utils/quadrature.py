'''
Various different quadrature rules for computing the integral of a function over a polygonal element.
'''

import numpy as np

## QUADRATURE RULES ##
def center_of_gravity_quadrature(func, element):
    return func(np.mean(element, axis=0))

def corner_quadrature(func, element):
    n = len(element)
    return 1/n * np.sum([func(element[i]) for i in range(n)])

def midpoint_quadrature(func, element):
    n = len(element)
    return 1/n * np.sum([func((element[i] + element[(i+1)%n])/2) for i in range(n)])

def trapezoid_quadrature(func, element):
    n = len(element)
    return 1/n * np.sum([func(element[i]) + func(element[(i+1)%n]) for i in range(n)])

def simpsons_quadrature(func, element):
    n = len(element)
    return 1/(3*n) * np.sum([func(element[i]) + 4*func((element[i] + element[(i+1)%n])/2) + func(element[(i+1)%n]) for i in range(n)])
