import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import ADASYN
from bayes_opt import BayesianOptimization
import numpy as np

# 加载数据和预处理
data = pd.read_csv('Dataset.csv', header=None)

# 特征和标签分离
# X = data.iloc[:, :6]  #no res
X = data.iloc[:, :6]  #res
yw = data.iloc[:, 8].astype(int) - 2  # 标签处理
yr = data.iloc[:, 9].astype(int) - 2

# yw = yr

# adasyn = ADASYN(sampling_strategy='minority')
# X_resampled, y_resampled = adasyn.fit_resample(X, yw)
X_resampled=X
y_resampled=yw
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=8)

# 定义贝叶斯优化函数
def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'random_state': 5
    }
    rf = RandomForestClassifier(**params)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# 设置超参数的搜索空间
params = {
    'n_estimators': (10, 100),
    'max_depth': (3, 10),
    'min_samples_split': (10, 20),
    'min_samples_leaf': (5, 15)
}

# 运行贝叶斯优化
rf_bo = BayesianOptimization(rf_evaluate, params, random_state=5)
rf_bo.maximize(init_points=10, n_iter=50)

# 使用最优参数训练模型
best_params = rf_bo.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
rf_best = RandomForestClassifier(**best_params)
rf_best.fit(X_train, y_train)

# 模型评估
y_pred = rf_best.predict(X_test)
# 转换y_test为二进制形式以适应mAP的计算
y_test_bin = label_binarize(y_test, classes=np.unique(y_resampled))
y_pred_bin = label_binarize(y_pred, classes=np.unique(y_resampled))

# 计算mAP
average_precision = dict()
for i in range(y_test_bin.shape[1]):
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_bin[:, i])
mAP = np.mean(list(average_precision.values()))

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"mAP: {mAP}")

