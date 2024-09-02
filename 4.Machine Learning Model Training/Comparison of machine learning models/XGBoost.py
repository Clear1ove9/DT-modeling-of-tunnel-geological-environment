import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

# 加载数据集
file_path = 'Dataset.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :6]  # 前6列是输入特征
y_water = data.iloc[:, 8]  # 第9列是water类别

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_water, test_size=0.3, random_state=33)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义贝叶斯优化函数
def catboost_evaluate(depth, learning_rate, l2_leaf_reg, iterations):
    params = {
        'depth': int(depth),
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'iterations': int(iterations),
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 5,
        'logging_level': 'Silent',
        'allow_writing_files': False
    }
    catboost = CatBoostClassifier(**params)
    cv_scores = cross_val_score(catboost, X_train_scaled, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)

# 设置超参数的搜索空间
params = {
    'depth': (3, 10),
    'learning_rate': (0.01, 1),
    'l2_leaf_reg': (1, 10),
    'iterations': (50, 300)
}

# 运行贝叶斯优化
catboost_bo = BayesianOptimization(catboost_evaluate, params, random_state=5)
catboost_bo.maximize(init_points=10, n_iter=50)

# 使用最优参数训练模型
best_params = catboost_bo.max['params']
best_params['depth'] = int(best_params['depth'])
best_params['iterations'] = int(best_params['iterations'])
catboost_best = CatBoostClassifier(**best_params)
catboost_best.fit(X_train_scaled, y_train)

# 模型评估
y_pred = catboost_best.predict(X_test_scaled)
y_test_binarized = label_binarize(y_test, classes=np.unique(y_water))

# 获取预测概率
y_score = catboost_best.predict_proba(X_test_scaled)

# 计算每个类别的平均精确度(AP)
average_precision = dict()
for j in range(y_test_binarized.shape[1]):
    average_precision[j] = average_precision_score(y_test_binarized[:, j], y_score[:, j])

# 计算mAP
mAP = np.mean(list(average_precision.values()))

# 存储并打印结果
results = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='macro'),
    'Recall': recall_score(y_test, y_pred, average='macro'),
    'F1 Score': f1_score(y_test, y_pred, average='macro'),
    'Mean Average Precision (mAP)': mAP
}

print("\nModel Performance:")
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")
