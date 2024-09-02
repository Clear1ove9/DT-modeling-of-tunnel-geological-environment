import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import ADASYN
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=5)

# 定义贝叶斯优化函数
def svm_evaluate(C, gamma):
    model = OneVsRestClassifier(SVC(C=C, gamma=gamma, kernel='rbf', probability=True, random_state=5))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# 设置超参数的搜索空间
params = {
    'C': (0.001, 100),
    'gamma': (0.0001, 1)
}

# 运行贝叶斯优化
svm_bo = BayesianOptimization(svm_evaluate, params, random_state=5)
svm_bo.maximize(init_points=10, n_iter=50)

# 使用最优参数训练模型
best_params = svm_bo.max['params']
svm_best = OneVsRestClassifier(SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', probability=True, random_state=5))
svm_best.fit(X_train, y_train)

# 为了计算mAP, 对y_test进行二值化
y_test_binarized = label_binarize(y_test, classes=np.unique(y_resampled))

# 使用训练好的模型对测试集进行预测，获取决策函数值
y_score = svm_best.predict_proba(X_test)

# 计算每个类别的平均精确度(AP)
average_precision = dict()
for i in range(y_test_binarized.shape[1]):
    average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])

# 计算mAP
mAP = np.mean(list(average_precision.values()))

# 打印原有的评价指标和mAP
accuracy = accuracy_score(y_test, svm_best.predict(X_test))
precision = precision_score(y_test, svm_best.predict(X_test), average='macro')
recall = recall_score(y_test, svm_best.predict(X_test), average='macro')
f1 = f1_score(y_test, svm_best.predict(X_test), average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"mAP: {mAP}")