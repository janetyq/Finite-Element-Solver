
# 2d vector operations
def calc_dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]

def calc_norm(vec):
    return (vec[0]**2 + vec[1]**2)**0.5

def calc_cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]