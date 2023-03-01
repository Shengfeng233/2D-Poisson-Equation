from pressure_utility import *


if __name__ == "__main__":
    # basic parameters
    origin_file = './xyuv.mat'
    start_index = 0
    end_index = 400
    resolution = 512
    filename_save = './p.mat'
    p_store = np.zeros((end_index-start_index, resolution, resolution))
    p_extend = generate_extend_grid(np.random.rand(resolution, resolution))  # initial guess
    delta = 2*np.pi/resolution
    # read data at a certain snapshot
    for step in range(start_index, end_index):
        x_unique, y_unique, t, u_mat, v_mat = read_data_at_snapshot(origin_file, step, resolution)
        # calculate the source term
        f = calculate_source(u_mat, v_mat, delta)
        f_extend = generate_extend_grid(f)
        p_mat, p_extend = solve_poisson_lu_sgs(p_extend, f_extend, delta, tol=1e-6, max_iter=1000)
        p_store[step, :, :] = p_mat
        # plot_solution(x_unique, y_unique, p_mat)
        print("Current step: ", step+1, "  Total step:", end_index-start_index)
    scipy.io.savemat(filename_save, {'p': p_store})
