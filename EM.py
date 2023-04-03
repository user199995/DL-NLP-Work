import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义高斯分布的参数
mean1, std1 = 164, 3
mean2, std2 = 176, 5

# 从两个高斯分布中生成各50个样本
data1 = np.random.normal(mean1, std1, 500)
data2 = np.random.normal(mean2, std2, 1500)
data = np.concatenate((data1, data2), axis=0)

# 将数据集按照 7:3 的比例分成训练集和测试集
np.random.shuffle(data)
train_data = data[:1500]
test_data = data[1500:]

# 定义EM算法的参数
w1 = 0.5
w2 = 0.5
max_iter = 100
epsilon = 1e-5

# 开始EM算法
for i in range(max_iter):
    # E步骤
    p1 = w1 * np.exp(-0.5 * (train_data - mean1) ** 2 / std1 ** 2) / np.sqrt(2 * np.pi * std1 ** 2)
    p2 = w2 * np.exp(-0.5 * (train_data - mean2) ** 2 / std2 ** 2) / np.sqrt(2 * np.pi * std2 ** 2)
    p = p1 + p2
    p1 = p1 / p
    p2 = p2 / p

    # M步骤
    w1_new = np.mean(p1)
    w2_new = np.mean(p2)
    mean1_new = np.sum(p1 * train_data) / np.sum(p1)
    mean2_new = np.sum(p2 * train_data) / np.sum(p2)
    std1_new = np.sqrt(np.sum(p1 * (train_data - mean1_new) ** 2) / np.sum(p1))
    std2_new = np.sqrt(np.sum(p2 * (train_data - mean2_new) ** 2) / np.sum(p2))

    # 判断是否收敛
    if abs(w1_new - w1) < epsilon and abs(w2_new - w2) < epsilon \
            and abs(mean1_new - mean1) < epsilon and abs(mean2_new - mean2) < epsilon \
            and abs(std1_new - std1) < epsilon and abs(std2_new - std2) < epsilon:
        break

    # 更新参数
    w1 = w1_new
    w2 = w2_new
    mean1 = mean1_new
    mean2 = mean2_new
    std1 = std1_new
    std2 = std2_new

# 计算模型在测试集上的对数似然
p1 = w1 * np.exp(-0.5 * (test_data - mean1) ** 2 / std1 ** 2) / np.sqrt(2 * np.pi * std1 ** 2)
p2 = w2 * np.exp(-0.5 * (test_data - mean2) ** 2 / std2 ** 2) / np.sqrt(2 * np.pi * std2 ** 2)
log_likelihood = np.sum(np.log(p1 + p2))

#打印模型在测试集上的对数似然值
print('Log likelihood on test data:', log_likelihood)

# 打印估计得到的高斯分布的参数
print('Gaussian distribution 1:')
print('Mean: {:.2f}'.format(mean1))
print('Standard deviation: {:.2f}'.format(std1))
print('Weight: {:.2f}'.format(w1))
print('Gaussian distribution 2:')
print('Mean: {:.2f}'.format(mean2))
print('Standard deviation: {:.2f}'.format(std2))
print('Weight: {:.2f}'.format(w2))