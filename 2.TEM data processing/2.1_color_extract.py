import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi import mlab


def read_cb(colorbar_img, colorbar_num, colorbar_max, colorbar_min):
    resistivity_range = np.linspace(colorbar_max, colorbar_min, colorbar_num)
    colorbar_height = colorbar_img.shape[0]
    strip_height = colorbar_height // colorbar_num
    
    strip_colors = []
    for i in range(colorbar_num):
        y1 = i * strip_height
        y2 = y1 + strip_height
        strip_img = colorbar_img[y1:y2, :, :]
        strip_color = np.mean(strip_img, axis=(0, 1))
        strip_colors.append(strip_color)
    return strip_colors, resistivity_range

def read_img(H_up30, strip_colors, resistivity_range):
    H_up30 = H_up30.astype(np.float32)
    H_up30_values = np.zeros(H_up30.shape[:2])
    for i in range(H_up30.shape[0]):
        for j in range(H_up30.shape[1]):
            if(np.all(H_up30[i,j] == [255, 255, 255])):
                H_up30_values[i, j] =np.nan
            else:
                color = H_up30[i, j, :]
                color_diffs = np.sum(np.abs(color - strip_colors), axis=1)
                resistivity_index = np.argmin(color_diffs)
                H_up30_values[i, j] = resistivity_range[resistivity_index]
    return H_up30_values  

# colorbar_up30 = cv2.imread("H_up30_cb1.png")
# H_up30 = cv2.imread("H_up30.png")

colorbar_H = cv2.imread('H_cb1.png')
H = cv2.imread('H.png')

# colorbar_down30 = cv2.imread('H_down30_cb1.png')
# H_down30 = cv2.imread('H_down30.png')

colorbar_V = cv2.imread('V_cb1.png')
V = cv2.imread('V.png')

# strip_colors_up30, resistivity_range_up30= read_cb(colorbar_up30, 14, 56, 14)  
# H_up30_values=read_img(H_up30, strip_colors_up30, resistivity_range_up30) 

strip_colors_H, resistivity_range_H= read_cb(colorbar_H, 35, 860, 520)  
H_values=read_img(H, strip_colors_H, resistivity_range_H) 

# strip_colors_down30, resistivity_range_down30= read_cb(colorbar_down30, 16, 46, 10)  
# H_down30_values=read_img(H_down30, strip_colors_down30, resistivity_range_down30) 

strip_colors_V, resistivity_range_V= read_cb(colorbar_V, 35, 860, 520)  
V_values=read_img(V, strip_colors_V, resistivity_range_V) 


# np.save('H_down30_values.npy', H_down30_values)
np.save('H_values.npy', H_values)
# np.save('H_up30_values.npy', H_up30_values)
np.save('V_values.npy', V_values)



