import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def read_data_at_snapshot(filename, num_snap, resolution):
    data_mat = scipy.io.loadmat(filename)
    x = data_mat['x']  # N*1
    y = data_mat['y']  # N*1
    t = data_mat['t'][num_snap, 0]  # T*1
    u = data_mat['u'][:, num_snap]  # N*T
    v = data_mat['v'][:, num_snap]  # N*T
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    t = t.item()
    u_mat = u.reshape(resolution, resolution)
    v_mat = v.reshape(resolution, resolution)
    return x_unique, y_unique, t, u_mat, v_mat


def read_data_all(filename, resolution):
    data_mat = scipy.io.loadmat(filename)
    x = data_mat['x']  # N*1
    y = data_mat['y']  # N*1
    t = data_mat['t'][:, 0]  # T*1
    u = data_mat['u'][:, :]  # N*T
    v = data_mat['v'][:, :]  # N*T
    x_unique = np.unique(x).reshape(-1, 1)
    y_unique = np.unique(y).reshape(-1, 1)
    return x_unique, y_unique, t, u, v


# generate grid with one extra virtual point along each direction
def generate_extend_grid(original_mat):
    resolution = original_mat.shape[0]
    extended_mat = np.ones((resolution+2, resolution+2))
    extended_mat[1: 1+resolution, 1:1+resolution] = original_mat
    extended_mat[1: 1+resolution, 0] = original_mat[:, -1]
    extended_mat[1: 1 + resolution, -1] = original_mat[:, 0]
    extended_mat[0, 1: 1 + resolution] = original_mat[-1, :]
    extended_mat[-1, 1: 1 + resolution] = original_mat[0, :]
    return extended_mat


# get the numerical derivative with 2 order central difference
def get_derivative(extended_mat, delta):
    resolution = extended_mat.shape[0]-2
    dx = (extended_mat[2:, 1:1+resolution] - extended_mat[:-2, 1:1+resolution]) / (2 * delta)
    dy = (extended_mat[1:1+resolution, 2:] - extended_mat[1:1+resolution, :-2]) / (2 * delta)
    return dx, dy


def calculate_source(u_mat, v_mat, delta):
    # generate grid for u and v
    u_extended = generate_extend_grid(u_mat)
    v_extended = generate_extend_grid(v_mat)
    ux, uy = get_derivative(u_extended, delta)
    vx, vy = get_derivative(v_extended, delta)
    f = -(ux*ux+2*uy*vx+vy*vy)
    return f


def assign_periodic_condition(extend_mat):
    extend_mat[0, :] = extend_mat[-2, :]
    extend_mat[-1, :] = extend_mat[1, :]
    extend_mat[:, 0] = extend_mat[:, -2]
    extend_mat[:, -1] = extend_mat[:, 1]
    return extend_mat


def plot_solution(x, y, p):
    mesh_x, mesh_y = np.meshgrid(x, y)
    plt.figure(figsize=(8, 6))
    plt.contourf(mesh_x, mesh_y, p, levels=200, cmap='jet')
    plt.show()
    return


def solve_poisson_sgs(p0, f_extend, delta, tol=1e-5, max_iter=10000):
    """
    solve poisson equation with source term, with periodic boundary condition
    :param p0: initial guess
    :param f_extend: source term
    :param delta: delta=dx=dy
    :param tol: the minimum tolerance
    :param max_iter: max acceptable number of iterations
    :return: return the solution (without extension of virtual grid), shape[resolution, resolution]
    """
    resolution = p0.shape[0]-2
    for it in range(max_iter):
        p_old = np.copy(p0)
        # upward scan
        for j in range(1, 1 + resolution):
            for i in range(1, 1 + resolution):
                p0[i, j] = 0.25 * (p0[i + 1, j] + p0[i - 1, j] + p0[i, j + 1] + p0[i, j - 1] - (delta ** 2) * f_extend[i, j])
        # downward scan
        for j in range(1, 1 + resolution):
            jr = resolution - j + 1
            for i in range(1, 1 + resolution):
                ir = resolution - i + 1
                p0[ir, jr] = 0.25 * (p0[ir+1, jr] + p0[ir-1, jr] + p0[ir, jr+1] + p0[ir, jr-1] - (delta ** 2) * f_extend[ir, jr])
        # assign periodic boundary condition
        p0 = assign_periodic_condition(p0)
        # Exit condition
        res = np.max(np.abs(p0 - p_old))
        if res < tol:
            break
        else:
            print("Current residual: ", res, "  Current iter num:", it)
    p_inside = p0[1:1+resolution, 1:1+resolution]
    return p_inside


def solve_poisson_lu_sgs(p0, f_extend, delta, tol=1e-6, max_iter=1000):
    """
    solve poisson equation with source term, with periodic boundary condition
    :param p0: initial guess
    :param f_extend: source term
    :param delta: delta=dx=dy
    :param tol: the minimum tolerance
    :param max_iter: max acceptable number of iterations
    :return: return the solution (without extension of virtual grid), shape[resolution, resolution]
    """
    resolution = p0.shape[0]-2
    for it in range(max_iter):
        p_old = np.copy(p0)
        # upward scan
        for j in range(1, 1 + resolution):
            for i in range(1, 1 + resolution):
                p0[i, j] = 0.25 * (p0[i - 1, j] + p0[i, j - 1] - (delta ** 2) * f_extend[i, j])
        # downward scan
        for j in range(1, 1 + resolution):
            jr = resolution - j + 1
            for i in range(1, 1 + resolution):
                ir = resolution - i + 1
                p0[ir, jr] = 0.25 * (p0[ir+1, jr] + p0[ir, jr+1]) + p0[ir, jr]
        # assign periodic boundary condition
        p0 = assign_periodic_condition(p0)
        # Exit condition
        res = np.max(np.abs(p0 - p_old))
        if res < tol:
            break
        else:
            print("Current residual: ", res, "  Current iter num:", it)
    p_inside = p0[1:1+resolution, 1:1+resolution]
    return p_inside, p0
