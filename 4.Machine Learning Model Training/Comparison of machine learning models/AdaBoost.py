import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import ADASYN
import numpy as np

#*********************************** 数据加载和预处理 ***********************************
# 加载数据和预处理
file_path = 'Dataset.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :6]  # 前7列是输入特征
yw = data.iloc[:, 8]  # 第10列是water类别

X_resampled=X
y_resampled=yw
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=5)

# 贝叶斯优化函数定义
def adaboost_evaluate(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf):
    dt = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf)
    )
    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'base_estimator': dt,
        'random_state': 5
    }
    model = AdaBoostClassifier(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# 贝叶斯优化
params = {
    'n_estimators': (20, 100),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 20),
    'min_samples_split': (10, 20),
    'min_samples_leaf': (5, 15)
}

adaboost_bo = BayesianOptimization(adaboost_evaluate, params, random_state=5)
adaboost_bo.maximize(init_points=10, n_iter=100)

# 训练最优模型
best_params = adaboost_bo.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['base_estimator'] = DecisionTreeClassifier(
    max_depth=int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf'])
)

# 删除不再需要的参数
del best_params['max_depth'], best_params['min_samples_split'], best_params['min_samples_leaf']

adaboost_best = AdaBoostClassifier(**best_params)
adaboost_best.fit(X_train, y_train)

# 模型评估
y_pred = adaboost_best.predict(X_test)
y_test_binarized = label_binarize(y_test, classes=np.unique(y_resampled))
y_score = adaboost_best.decision_function(X_test)

# 计算每个类别的平均精确度(AP)
average_precision = dict()
for i in range(y_test_binarized.shape[1]):
    average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])

# 计算mAP
mAP = np.mean(list(average_precision.values()))

# 打印评价指标和mAP
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"mAP: {mAP}")