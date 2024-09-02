import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab
import math
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from pykrige.uk import UniversalKriging
from scipy.interpolate import RegularGridInterpolator
from pykrige.ok3d import OrdinaryKriging3D
from scipy.ndimage import zoom
import pandas as pd

def plotall(H_values,V_values):
    # 创建一个4x2的子图布局
    fig, axs = plt.subplots(2, 2)

    # 在每个子图中绘制二维数组并显示colorbar
    im1 = axs[0, 0].imshow(H_values, cmap='jet')
    fig.colorbar(im1, ax=axs[0, 0])

    im4 = axs[1, 1].imshow(V_values, cmap='jet')
    fig.colorbar(im4, ax=axs[1, 1])

    # 调整子图布局
    fig.tight_layout()

    # 显示图像
    plt.show()

# H_up30_values = np.load('H_up30_values.npy')
H_values = np.load('H_values.npy')
# H_down30_values = np.load('H_down30_values.npy')
V_values = np.load('V_values.npy')

width_h = 55 
height_h = 40

width_v = 55
height_v = 40


H_values = cv2.resize(H_values, (width_v, height_v),interpolation=cv2.INTER_LINEAR)
V_values = cv2.resize(V_values, (width_v, height_v),interpolation=cv2.INTER_LINEAR)

plotall(H_values,V_values)

grid = np.empty((V_values.shape[1], H_values.shape[1], H_values.shape[0]), order='F')
grid[:]=np.nan

d1=V_values.shape[1]
d2=H_values.shape[1]
d3=H_values.shape[0]
alpha=[np.pi/6,np.pi/12,0,-np.pi/12, -np.pi/6 ]
for i in range(H_values.shape[0]):
    for j in range(H_values.shape[1]):
            # newx_up30 = math.floor(d1/2-np.sin(alpha[0])*(d3-i))
            # newy_up30 = j
            # newz_up30 = math.floor(d3-i*np.cos(alpha[0]))-1 
            # grid[newx_up30,newy_up30,newz_up30]=H_up30_values[i,j]
            
            newx_0 = math.floor(d1/2-np.sin(alpha[2])*(d3-i))
            newy_0 = j
            newz_0 = math.floor(d3-i*np.cos(alpha[2])) -1 
            grid[newx_0,newy_0,newz_0]=H_values[i,j]   

            # newx_down30 = math.floor(d1/2-np.sin(alpha[4])*(d3-i))
            # newy_down30 = j
            # newz_down30 = math.floor(d3-i*np.cos(alpha[4]))-1 
            # # print(newz_down30)
            # grid[newx_down30,newy_down30,newz_down30]=H_down30_values[i,j] 
        
for i in range(V_values.shape[0]):
    for j in range(V_values.shape[1]): 
            newx_v = j
            newy_v = math.floor(d2/2)
            newz_v = math.floor(d3-i)-1 
            # print(newx_v,newy_v,newz_v)
            grid[newx_v,newy_v,newz_v]=V_values[i,j]  
            
            
grid[np.isnan(grid)] = 0

# scale_factors = (1.01, 0.99, 1)
# grid = zoom(grid, scale_factors)
print(grid.shape)
# np.save('b_grid.npy', grid)

# 定义结构化数组的字段名称和数据类型
data = np.zeros(grid.shape[0]*grid.shape[1]*grid.shape[2], dtype=[('x', np.float), ('y', np.float), ('z', np.float), ('res', np.float)])
# 遍历数组并对其进行赋值
count=0
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        for k in range(grid.shape[2]):
            # 访问第 i, j, k 个元素
            data[count]['x'] = k  # 对 'x' 字段赋值
            data[count]['y'] = (grid.shape[1]-1)/2-j  # 对 'y' 字段赋值
            data[count]['z'] = (grid.shape[0]-1)/2-i  # 对 'z' 字段赋值
            data[count]['res'] = grid[i, j, k]  # 对 'temperature' 字段赋值
            count=count+1
            # print(count)
np.save('TEM_grid.npy', data)

output_columns = ['X', 'Y', 'Z', 'res']
output_file = pd.DataFrame({'X': data['x'], 'Y': data['y'], 'Z': data['z'], 'res': data['res']})
output_file.to_csv('Raw_res_data.csv')

# 创建一个布尔索引数组，表示哪些点的温度不为0
nonzero_temperature_indices = data['res'] != 0

# 创建一个3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用非零温度的索引来选择要绘制的点
scatter =ax.scatter(data['x'][nonzero_temperature_indices],
            data['y'][nonzero_temperature_indices],
            data['z'][nonzero_temperature_indices],
            c=data['res'][nonzero_temperature_indices], cmap='jet')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.colorbar(scatter)
plt.show()











