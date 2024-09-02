import numpy as np
import pandas as pd
from mayavi import mlab
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
import joblib  # 用于加载已保存的高斯过程模型
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import trimesh
from stl import mesh
from skimage import measure
import mcubes
from imblearn.over_sampling import ADASYN
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from vtk.util.numpy_support import numpy_to_vtk
import vtk

def pointgrid(x,y,z,Vp):
    x_min, x_max = int(x.min()), int(x.max())
    y_min, y_max = int(y.min()), int(y.max())
    z_min, z_max = int(z.min()), int(z.max())
    tsp_grid = np.full((int(x_max-x_min)+1, int(y_max-y_min)+1, int(z_max-z_min)+1), np.nan)

    for x_i, y_i, z_i, res_i in zip(x.astype(int), y.astype(int), z.astype(int), Vp):
        tsp_grid[int(x_i-x_min), int(y_i-y_min), int(z_i-z_min)] = res_i
    return tsp_grid

# 加载高斯过程分类模型
w_model = joblib.load('Water_intensity_TSP&TEM.model')
r_model = joblib.load('Rock_integrity_TSP&TEM.model')

wfill_model = joblib.load('Water_intensity_TSP.model')
rfill_model = joblib.load('Rock_integrity_TSP.model')

w_scaler = joblib.load('Water_intensity_TSP&TEM_scaler.model')
r_scaler = joblib.load('Rock_integrity_TSP&TEM_scaler.model')

wfill_scaler = joblib.load('Water_intensity_TSP_scaler.model')
rfill_scaler = joblib.load('Rock_integrity_TSP_scaler.model')

# 加载预测数据
data = np.load("Fused_data.npy")

Xc = data['x']
Yc = data['y']
Zc = data['z']
Vp = data['Vp']
Vs = data['Vs']
res = data['res']
ro = data['ro']
E = data['E']/1e10
Pr = data['Pr']


w = np.empty_like(res)
w[:] = np.nan  # 将所有元素初始化为nan
non_nan_indices = ~np.isnan(res)
nan_indices = np.isnan(res)
r = np.empty_like(res)
r[:] = np.nan  # 将所有元素初始化为nan 


N_res = (res - np.nanmin(res)) / (np.nanmax(res) - np.nanmin(res))
x_pre0 = np.column_stack((Vp, Vs, ro, E, Pr, N_res))
x_pre = x_pre0[non_nan_indices]

x_prefill0 = np.column_stack((Vp, Vs, ro, E, Pr))
x_prefill = x_prefill0[nan_indices]

# 使用高斯过程分类模型进行预测
x_pre_scaled1=w_scaler.transform(x_pre)
x_pre_scaled2=r_scaler.transform(x_pre)

x_prefill_scaled1=wfill_scaler.transform(x_prefill)
x_prefill_scaled2=rfill_scaler.transform(x_prefill)

w_pre1 = w_model.predict(x_pre_scaled1)
r_pre1 = r_model.predict(x_pre_scaled2)
w_prefill1 = wfill_model.predict(x_prefill_scaled1)
r_prefill1 = rfill_model.predict(x_prefill_scaled2)

w[non_nan_indices] = w_pre1
r[non_nan_indices] = r_pre1
w[nan_indices] = w_prefill1
r[nan_indices] = r_prefill1

output_struc = {
    'x': Xc,
    'y': Yc,
    'z': Zc,
    'w': w,
    'r': r
}
output = pd.DataFrame(output_struc)
output_file = 'output_data.csv'
output.to_csv(output_file)

w_grid = pointgrid(Xc, Yc, Zc, w)
r_grid = pointgrid(Xc, Yc, Zc, r)

fig = mlab.figure(size=(500, 500))
volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(r_grid))
mlab.colorbar()
mlab.show()
