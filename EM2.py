import pandas as pd
from sklearn.mixture import GaussianMixture

# 加载数据
data = pd.read_csv("D:\\vscode script\\PYTHON\\sklearn\\HeightWeight.csv")

# 选择特征：身高和体重
X = data[['Height(cm)', 'Weight(kg)']].values

# 初始化高斯混合模型，假设有两个组件（男性和女性）
gmm = GaussianMixture(n_components=2, random_state=42)

# 拟合模型
gmm.fit(X)

# 使用模型参数来估计隐变量的分布以及男女的身高体重分布参数
means = gmm.means_  # 各组件的均值
covariances = gmm.covariances_  # 各组件的协方差
weights = gmm.weights_  # 各组件的权重，可以理解为男性和女性的比例

# 输出结果
print("均值：\n", means)
print("协方差：\n", covariances)
print("权重（男女比例）：\n", weights)
