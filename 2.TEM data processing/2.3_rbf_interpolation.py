import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
import pandas as pd

# 读取数据
tem = pd.read_csv('Raw_res_data.csv')
x_tem = tem['X'].values
y_tem = tem['Y'].values
z_tem = tem['Z'].values
v_tem = tem['res'].values

# 过滤掉 v_tem 为 0 的点
valid_indices = (v_tem != 0)
x = x_tem[valid_indices]
y = y_tem[valid_indices]
z = z_tem[valid_indices]
temperature = v_tem[valid_indices]

# 构建输入数据点和凸包
points = np.column_stack((x, y, z))
hull = ConvexHull(points)

# 使用RBFInterpolator进行三维插值，仅对凸包内的点进行插值
rbf_interpolator = RBFInterpolator(points, temperature, kernel='gaussian', epsilon=1, neighbors=50)

# 创建用于插值的网格
grid_x, grid_y, grid_z = np.mgrid[0:39:40j, -27:27:55j, -27:27:55j]
grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

# 判断网格点是否在凸包内
from scipy.spatial import Delaunay
tri = Delaunay(points[hull.vertices])
inside = tri.find_simplex(grid_points) >= 0

# 初始化插值结果，凸包外的点赋值为 0
interpolated_temp = np.zeros(grid_points.shape[0])

# 仅对凸包内的点进行插值
interpolated_temp[inside] = rbf_interpolator(grid_points[inside])

# 将插值结果整合为 DataFrame
interpolated_data = pd.DataFrame({
    'x': grid_points[:, 0],
    'y': grid_points[:, 1],
    'z': grid_points[:, 2],
    'res': interpolated_temp
})
# 将 'res' 列中为 0 的值设置为 NaN
interpolated_data['res'].replace(0, np.nan, inplace=True)
# 导出为 CSV 文件
interpolated_data.to_csv('Interpolated_res_data.csv', index=False)



# 筛选出 interpolated_temp 不为 0 的点
non_zero_indices = interpolated_temp != 0
filtered_grid_points = grid_points[non_zero_indices]
filtered_interpolated_temp = interpolated_temp[non_zero_indices]

# 可视化插值结果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制插值后的非零点的散点图
scatter_interp = ax.scatter(
    filtered_grid_points[:, 0], 
    filtered_grid_points[:, 1], 
    filtered_grid_points[:, 2], 
    c=filtered_interpolated_temp, 
    cmap='jet', 
)

ax.set_title('Interpolated Data with Convex Hull Masking')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加颜色条
fig.colorbar(scatter_interp, ax=ax, shrink=0.5, aspect=5, label='res')

plt.show()
