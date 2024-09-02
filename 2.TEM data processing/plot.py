import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
tem = pd.read_csv('Interpolated_res_data.csv')
x_tem = tem['x'].values
y_tem = tem['y'].values
z_tem = tem['z'].values
v_tem = tem['res'].values

# 创建三维散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
scatter = ax.scatter(x_tem, y_tem, z_tem, c=v_tem, cmap='jet', marker='o')

# 添加颜色条
colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
colorbar.set_label('Temperature')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
ax.set_title('3D Scatter Plot of Temperature Distribution')

# 显示图像
plt.show()
