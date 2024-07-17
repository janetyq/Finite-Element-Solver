import numpy as np
import igl

def get_handle_transforms(orig_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
    handle_transforms = []
    for index in range(len(new_positions)):
        translation = new_positions[index] - orig_positions[index]  # transformation
        handle_transforms.append([1, 0, 0, 1, translation[0], translation[1]])
    return np.array(handle_transforms).reshape(len(new_positions) * 3, 2)

def linear_weights(vertex_positions: np.ndarray, handle_positions: np.ndarray) -> np.ndarray:
    """
    Compute linear weights for vertices on a triangular mesh

    Params:
        * vertex_positions: (Nx2) array
        * handle_positions: (Hx2) array

    Return value:
        * weights: (NxH) array
    """
    num_vertices, num_handles = vertex_positions.shape[0], handle_positions.shape[0]
    weights = np.zeros((num_vertices, num_handles))

    for h_idx, handle_position in enumerate(handle_positions):
        for v_idx, vertex_position in enumerate(vertex_positions):
            d = np.linalg.norm(vertex_position - handle_position)**0.5
            weights[v_idx, h_idx] = 1/d if d > 1e-14 else np.finfo(np.float64).max

    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights

def bbw_weights(vertex_positions: np.ndarray, elements: np.ndarray, handle_positions: np.ndarray) -> np.ndarray:
    """
    Compute BBW weights for vertices on a triangular mesh

    Params:
        * vertex_positions: (Nx2) array
        * elements: (Fx3) array
        * handle_positions: (Hx2) array

    Return value:
        * weights: (NxH) array
    """
    _, b, bc = igl.boundary_conditions(
        np.hstack((vertex_positions, np.zeros((vertex_positions.shape[0], 1)))), 
        elements.astype(np.int64), 
        np.hstack((handle_positions, np.zeros((handle_positions.shape[0], 1)))).astype(np.double), 
        np.arange(len(handle_positions), dtype=np.int64), 
        np.zeros((0, 2), dtype=np.int64),
        np.zeros((0, 2), dtype=np.int64),
        np.zeros((0, 3), dtype=np.int64))
    bbw = igl.pyigl_classes.BBW(0, 100)  # type: ignore
    weights = bbw.solve(vertex_positions, elements, b, bc)
    weights /= weights.sum(axis=1, keepdims=True)
    print(f"BBW weights: {weights.shape}")
    return weights


def generate_rect_mesh(width: int, height: int, num_x: int, num_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.zeros((num_x * num_y, 2))
    elements = np.zeros((2 * (num_x - 1) * (num_y - 1), 3), dtype=int)
    boundary_indices = np.zeros(num_x * num_y, dtype=int)

    for i in range(num_x):
        for j in range(num_y):
            idx = i * num_y + j
            vertices[idx] = [i * width / (num_x - 1), j * height / (num_y - 1)]

    element_idx = 0
    for i in range(num_x - 1):
        for j in range(num_y - 1):
            if (i+j) % 2 == 0:
                idx = i * num_y + j
                elements[element_idx] = [idx, idx + 1, idx + num_y]
                elements[element_idx + 1] = [idx + 1, idx + num_y + 1, idx + num_y]
                element_idx += 2
            else:
                idx = i * num_y + j
                elements[element_idx] = [idx, idx + 1, idx + num_y + 1]
                elements[element_idx + 1] = [idx, idx + num_y + 1, idx + num_y]
                element_idx += 2

    boundary_edges = calculate_boundary_edges(vertices, elements)
    boundary_indices = np.unique(boundary_edges.flatten())

    return vertices, elements, boundary_indices, boundary_edges

def calculate_boundary_edges(vertices, elements):
    edges = set()
    boundary_edges = set()

    # Create edges from each element
    for element in elements:
        for i in range(len(element)):
            edge = tuple(sorted([element[i], element[(i + 1) % len(element)]]))
            if edge in edges:
                boundary_edges.discard(edge)
            else:
                edges.add(edge)
                boundary_edges.add(edge)

    return np.array(list(boundary_edges))

if __name__ == "__main__": # weird demo of moving one handle and seeing mesh deform
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    vertices, elements, boundary, BE = generate_rect_mesh(1, 1, 5, 5)

    control_handles = []
    n = 3
    for i in range(n):
        control_handles.append([i/n, 0.0]) # bottom
        control_handles.append([(i+1)/n, 1.0]) # top
        control_handles.append([0.0, (i+1)/n]) # left
        control_handles.append([1.0, i/n]) # right
    control_handles = np.array(control_handles, dtype=np.float64)

    fixed_vertices = [idx for idx in range(len(vertices)) if vertices[idx][1] == 0]

    weights = bbw_weights(vertices, elements, control_handles)
    # weights = linear_weights(vertices, control_handles)
    handle_transforms = get_handle_transforms(control_handles, control_handles)

    lbs = igl.lbs_matrix(vertices, weights)  # type: ignore

    print(f"vertices: {vertices.shape}, W: {weights.shape}, lbs: {lbs.shape}")
    deformed_vertices = lbs @ handle_transforms

    new_control_handles = np.copy(control_handles)

    # Define a function to update the deformed mesh plot
    def update_plot(event):
        new_control_handles[0][0] -= 0.1
        handle_transforms = get_handle_transforms(control_handles, new_control_handles)
        deformed_vertices = lbs @ handle_transforms
        deformed_vertices[fixed_vertices] = vertices[fixed_vertices]
        axs[1].clear()  # control_handleslear the current plot
        axs[1].triplot(deformed_vertices[:, 0], deformed_vertices[:, 1], elements)
        axs[1].scatter(new_control_handles[:, 0], new_control_handles[:, 1], color='red')
        axs[1].set_title("Deformed Mesh")
        axs[1].set_aspect('equal', 'box')
        plt.draw()

    # control_handlesreate the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].triplot(vertices[:, 0], vertices[:, 1], elements)
    axs[0].scatter(vertices[boundary, 0], vertices[boundary, 1], color='blue', alpha=0.5, s=100)
    axs[0].scatter(control_handles[:, 0], control_handles[:, 1], color='red')
    axs[0].set_title("Original Mesh")
    axs[0].set_aspect('equal', 'box')
    axs[1].triplot(deformed_vertices[:, 0], deformed_vertices[:, 1], elements)
    axs[1].scatter(control_handles[:, 0], control_handles[:, 1], color='red')
    axs[1].set_title("Deformed Mesh")
    axs[1].set_aspect('equal', 'box')

    # control_handlesreate a button to update the plot
    update_button_ax = plt.axes([0.8, 0.05, 0.1, 0.05])  # [left, bottom, width, height]
    update_button = Button(update_button_ax, 'Update')
    update_button.on_clicked(update_plot)

    plt.show()

