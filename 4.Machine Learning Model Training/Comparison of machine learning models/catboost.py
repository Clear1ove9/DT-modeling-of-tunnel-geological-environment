import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from bayes_opt import BayesianOptimization


# 加载数据和预处理
file_path = 'Dataset.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :6]  # 前7列是输入特征
yw = data.iloc[:, 8]  # 第10列是water类别

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, yw, test_size=0.3, random_state=5)

# 使用Pool
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test)

# 定义贝叶斯优化函数
def catboost_evaluate(iterations, depth, learning_rate):
    params = {
        'iterations': int(iterations),  # 确保迭代次数为整数
        'depth': int(depth),  # 确保树的深度为整数
        'learning_rate': learning_rate,
        'loss_function': 'MultiClass',
        'random_seed': 5,
        'verbose': False,
        'early_stopping_rounds': 50  # 添加早停
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, verbose=False)
    preds = model.predict(test_pool)
    return accuracy_score(y_test, preds)

# 设置超参数的搜索空间
params = {
    'iterations': (100, 500),  # 减少迭代次数上限
    'depth': (4, 8),  # 减少最大深度
    'learning_rate': (0.01, 0.2)  # 缩小学习率范围
}

# 运行贝叶斯优化
catboost_bo = BayesianOptimization(catboost_evaluate, params, random_state=5)
catboost_bo.maximize(init_points=5, n_iter=50)  # 减少贝叶斯优化的迭代次数和初始点

# 使用最优参数训练模型
best_params = catboost_bo.max['params']
catboost_best = CatBoostClassifier(**best_params, loss_function='MultiClass', random_seed=5, verbose=False)
catboost_best.fit(train_pool)

# 进行预测
y_pred = catboost_best.predict(X_test)
y_pred_proba = catboost_best.predict_proba(X_test)

# 评估
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
average_precision = [average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i]) for i in range(y_test_binarized.shape[1])]
mAP = np.mean(average_precision)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nMap: {mAP}")
