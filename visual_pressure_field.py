from pressure_utility import *
import os
import imageio
import matplotlib as mpl
mpl.use("Agg")


def make_flow_gif(start_index, end_index, t_list, mode, name='q', fps_num=5):
    gif_images = []
    t_unique = np.unique(t_list).reshape(-1, 1)
    time_series = t_unique[start_index:end_index, 0].reshape(-1, 1)
    for select_time in time_series:
        time = select_time.item()
        gif_images.append(imageio.imread('./gif_make/' + 'time' + "{:.2f}".format(time) + ' p.png'))
    imageio.mimsave((mode + name + '.gif'), gif_images, fps=fps_num)



filename = './p.mat'
data = scipy.io.loadmat(filename)
pressure_mat = data['p']
origin_file = './xyuv.mat'
time_steps = pressure_mat.shape[0]
x_unique, y_unique, t_list, u_mat_all, v_mat_all = read_data_all(origin_file, resolution=512)
# mesh_x, mesh_y = np.meshgrid(x_unique, y_unique)
# for i in range(time_steps):
#     p_mat = pressure_mat[i, :, :]
#     plt.figure(figsize=(8, 6))
#     plt.contourf(mesh_x, mesh_y, p_mat, levels=200, cmap='jet')
#     if not os.path.exists('gif_make'):
#         os.makedirs('gif_make')
#     plt.savefig('./gif_make/' + 'time' + "{:.2f}".format(t_list[i]) + ' p.png')
#     print("Current step: ", i+1, "    Total step: ", time_steps)

start_index = 0
end_index = time_steps
make_flow_gif(start_index, end_index, t_list, mode='draw', name='p', fps_num=20)