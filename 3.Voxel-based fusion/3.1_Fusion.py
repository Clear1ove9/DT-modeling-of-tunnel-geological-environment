import numpy as np
import pandas as pd
from mayavi import mlab
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.spatial import distance_matrix
from scipy.ndimage import zoom


tspdata= pd.read_csv('TSP_data.csv')
def pointgrid(x,y,z,Vp):
    # 确定网格的范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    # 创建一个填充NaN的三维网格
    tsp_grid = np.full((int(x_max-x_min)+1, int(y_max-y_min+1), int(z_max-z_min)+1), np.nan)

    # 使用已知的电阻率值填充网格
    for x_i, y_i, z_i, res_i in zip(x, y, z, Vp):
        tsp_grid[int(x_i-x_min), int(y_i-y_min), int(z_i-z_min)] = res_i
    return tsp_grid

#TSP数据读取瑜保存
x_tsp=(tspdata['X'].values-2256)*(-1)
# print("x_tsp",x_tsp)
y_tsp=tspdata['Y'].values
z_tsp=tspdata['Z'].values

Vp_tsp=tspdata['Vp'].values
Vs_tsp=tspdata['Vs'].values
ro_tsp=tspdata['ro'].values
E_tsp=tspdata['E'].values
Pr_tsp=tspdata['Pr'].values  

tsp_grid_Vp0=pointgrid(x_tsp, y_tsp, z_tsp, Vp_tsp)
tsp_grid_Vs0=pointgrid(x_tsp, y_tsp, z_tsp, Vs_tsp)
tsp_grid_ro0=pointgrid(x_tsp, y_tsp, z_tsp, ro_tsp)
tsp_grid_E0=pointgrid(x_tsp, y_tsp, z_tsp, E_tsp)
tsp_grid_Pr0=pointgrid(x_tsp, y_tsp, z_tsp, Pr_tsp)

# scale_factors = (1, 1, 1)
# tsp_grid_Vp = zoom(tsp_grid_Vp0, scale_factors)
# tsp_grid_Vs = zoom(tsp_grid_Vs0, scale_factors)
tsp_grid_Vp =tsp_grid_Vp0
tsp_grid_Vs =tsp_grid_Vs0
tsp_grid_ro=tsp_grid_ro0
tsp_grid_E=tsp_grid_E0
tsp_grid_Pr=tsp_grid_Pr0


# mlab.clf()
# # mlab.contour3d(tsp_grid)
# volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(tsp_grid_Vp))
# mlab.colorbar()
# mlab.show()

#TEM数据读取与保存
tem_data =pd.read_csv('Interpolated_res_data.csv')
x_tem=tem_data.iloc[:,0].values
y_tem=tem_data.iloc[:,1].values
z_tem=tem_data.iloc[:,2].values
res=  tem_data.iloc[:,3].values

tem_grid=pointgrid(x_tem,y_tem,z_tem,res)

# mlab.clf()
# # mlab.contour3d(tsp_grid)
# volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(tem_grid))
# mlab.colorbar()
# mlab.show()

# TEM定义结构化数组的字段名称和数据类型
tem_data = np.zeros(tem_grid.shape[0]*tem_grid.shape[1]*tem_grid.shape[2], dtype=[('x', np.float), ('y', np.float), ('z', np.float), ('res', np.float)])
# 遍历数组并对其进行赋值
count=0
for i in range(tem_grid.shape[0]):
    for j in range(tem_grid.shape[1]):
        for k in range(tem_grid.shape[2]):
            # 访问第 i, j, k 个元素
            tem_data[count]['x'] = i  # 对 'x' 字段赋值
            tem_data[count]['y'] = j-(tem_grid.shape[1]-1)//2  # 对 'y' 字段赋值
            tem_data[count]['z'] = k-(tem_grid.shape[2]-1)//2  # 对 'z' 字段赋值
            tem_data[count]['res'] = tem_grid[i, j, k]  # 对 'res' 字段赋值
            count=count+1
            # print(count)
            
            
# TSP定义结构化数组的字段名称和数据类型
tsp_data = np.zeros(tsp_grid_Vp.shape[0]*tsp_grid_Vp.shape[1]*tsp_grid_Vp.shape[2], dtype=[('x', np.float), ('y', np.float), ('z', np.float), ('Vp', np.float), ('Vs', np.float), ('ro', np.float), ('E', np.float), ('Pr', np.float)])
# 遍历数组并对其进行赋值
count=0
for i in range(tsp_grid_Vp.shape[0]):
    for j in range(tsp_grid_Vp.shape[1]):
        for k in range(tsp_grid_Vp.shape[2]):
            # 访问第 i, j, k 个元素
            tsp_data[count]['x'] = i  # 对 'x' 字段赋值
            tsp_data[count]['y'] = j-(tsp_grid_Vp.shape[1]-1)//2  # 对 'y' 字段赋值
            tsp_data[count]['z'] = k-(tsp_grid_Vp.shape[2]-1)//2  # 对 'z' 字段赋值
            tsp_data[count]['Vp'] = tsp_grid_Vp[i, j, k]  # 对 'Vp' 字段赋值
            tsp_data[count]['Vs'] = tsp_grid_Vs[i, j, k]  # 对 'Vp' 字段赋值
            tsp_data[count]['ro'] = tsp_grid_ro[i, j, k]
            tsp_data[count]['E'] = tsp_grid_E[i, j, k]
            tsp_data[count]['Pr'] = tsp_grid_Pr[i, j, k]
            
            count=count+1
            # print(count)
            
#***********************************merge****************************************
# 从tsp_data和tem_data数据构建DataFrame
tsp_df = pd.DataFrame(tsp_data)
tem_df = pd.DataFrame(tem_data)

merge = np.zeros(tsp_grid_Vp.shape[0]*tsp_grid_Vp.shape[1]*tsp_grid_Vp.shape[2], dtype=[('x', np.float), ('y', np.float), ('z', np.float), ('Vp', np.float), ('Vs', np.float), ('ro', np.float), ('E', np.float), ('Pr', np.float) ,('res', np.float)])

count=0
for i in range(tsp_grid_Vp.shape[0]):
    for j in range(tsp_grid_Vp.shape[1]):
        for k in range(tsp_grid_Vp.shape[2]):
            # 访问第 i, j, k 个元素
            merge[count]['x'] = i  # 对 'x' 字段赋值
            merge[count]['y'] = j-(tsp_grid_Vp.shape[1]-1)//2  # 对 'y' 字段赋值
            merge[count]['z'] = k-(tsp_grid_Vp.shape[2]-1)//2  # 对 'z' 字段赋值
            merge[count]['Vp'] = tsp_grid_Vp[i, j, k]  
            merge[count]['Vs'] = tsp_grid_Vs[i, j, k]  
            merge[count]['ro'] = tsp_grid_ro[i, j, k]  
            merge[count]['E'] = tsp_grid_E[i, j, k] 
            merge[count]['Pr'] = tsp_grid_Pr[i, j, k]  
            
            if(i<10 or j>=tem_grid.shape[1] or k>=tem_grid.shape[2]):
                merge[count]['res']=np.nan
            else:
                merge[count]['res'] =tem_grid[i-10, j-1, k-1]
            count=count+1
            # print(count)

merge_df = pd.DataFrame(merge)


# 将合并后的DataFrame转为结构化数组
new_dtype = np.dtype([("x", float), ("y", float), ("z", float), ("Vp", float), ("Vs", float), ("ro", float),("E", float),("Pr", float),("res", float)])
new_data = np.zeros((merge_df.shape[0]), dtype=new_dtype)

new_data["x"] = merge_df["x"].values
new_data["y"] = merge_df["y"].values
new_data["z"] = merge_df["z"].values
new_data["Vp"] = merge_df["Vp"].values
new_data["Vs"] = merge_df["Vs"].values
new_data["ro"] = merge_df["ro"].values
new_data["E"] = merge_df["E"].values
new_data["Pr"] = merge_df["Pr"].values
new_data["res"] = merge_df["res"].values


# merged_df.to_csv('output_TEM.csv')
np.save("Fused_data.npy",new_data)

# # merged_df.to_csv('output_TEM.csv')
# new_grid1=pointgrid(new_data['x'], new_data['y'], new_data['z'], new_data['res'])
# # new_grid2=pointgrid(new_data['x'], new_data['y'], new_data['z'], new_data['res'])

# mlab.clf()
# # mlab.contour3d(tsp_grid)
# volume1 = mlab.pipeline.volume(mlab.pipeline.scalar_field(new_grid1))
# # volume2 = mlab.pipeline.volume(mlab.pipeline.scalar_field(new_grid2))
# mlab.colorbar()
# mlab.show()
