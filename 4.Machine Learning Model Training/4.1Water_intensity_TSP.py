import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib 
from imblearn.over_sampling import ADASYN

# 加载数据集
file_path = 'Dataset.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :5]  # 
Y = data.iloc[:, 8]  # 

adasyn = ADASYN(sampling_strategy='minority')

X_resampled, y_resampled = adasyn.fit_resample(X, Y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=41)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 定义核函数：RBF核 + 白噪声核
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

# 初始化高斯过程分类模型
gpc = GaussianProcessClassifier(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10, random_state=41)

# 训练模型
gpc.fit(X_train_scaled, y_train)

# 保存模型为.model格式
model_filename = 'Water_intensity_TSP.model'
joblib.dump(scaler, 'Water_intensity_TSP_scaler.model')

# 使用训练好的模型进行预测
y_pred = gpc.predict(X_test_scaled)
y_pred_proba = gpc.predict_proba(X_test_scaled)  # 获取预测概率，用于计算mAP

# 计算并输出性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 对测试集标签进行二值化处理，用于mAP的计算
n_classes = len(np.unique(y_resampled))  # 计算类别数
Y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

# 计算每个类别的AP，并计算mAP
average_precisions = []
for i in range(n_classes):
    average_precision = average_precision_score(Y_test_binarized[:, i], y_pred_proba[:, i])
    average_precisions.append(average_precision)

# 计算mAP
mAP = np.mean(average_precisions)


