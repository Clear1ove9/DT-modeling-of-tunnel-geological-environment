import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#***********************************Model Training*****************************************
data = pd.read_csv('test_train - 副本.csv', header=None)
# 划分特征和标签
X = data.iloc[:, :7] #res
# X = data.iloc[:, :6] #no res
yw = data.iloc[:, 7].astype(int)
yw=yw-2
yr = data.iloc[:, 8].astype(int)
yr=yr-2
yw=yr
# 初始化ADASYN模型
adasyn = ADASYN(sampling_strategy='minority')
Xw, yww = adasyn.fit_resample(X, yw)
# Xw, yww = (X, yw)  


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(Xw, yww, test_size=0.3, random_state=5)

# 初始化MinMaxScaler进行归一化处理
scaler = MinMaxScaler()

# 对训练和测试数据进行归一化
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义高斯过程分类器的贝叶斯优化函数
def gpc_evaluate(log_length_scale, log_noise_level):
    length_scale = np.exp(log_length_scale)
    noise_level = np.exp(log_noise_level)
    kernel = 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=5)
    cv_scores = cross_val_score(gpc, X_train_scaled, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# 设置超参数的搜索空间（对数空间）
params = {
    'log_length_scale': (np.log(0.1), np.log(10)),
    'log_noise_level': (np.log(1e-08), np.log(1)),
}

# 运行贝叶斯优化
gpc_bo = BayesianOptimization(gpc_evaluate, params, random_state=5)
gpc_bo.maximize(init_points=15, n_iter=50)

# 使用最优参数训练模型
best_params = gpc_bo.max['params']
best_length_scale = np.exp(best_params['log_length_scale'])
best_noise_level = np.exp(best_params['log_noise_level'])
kernel_best = 1.0 * RBF(length_scale=best_length_scale) + WhiteKernel(noise_level=best_noise_level)
gpc_best = GaussianProcessClassifier(kernel=kernel_best, random_state=5)
gpc_best.fit(X_train_scaled, y_train)

# 模型评估
y_pred = gpc_best.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")