import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# 加载数据
data = pd.read_csv("D:\\vscode script\\PYTHON\\sklearn\\HeightWeight.csv")

# 打乱数据10次
# for _ in range(10):
#     data = data.sample(frac=1).reset_index(drop=True)

# 初始化参数
pi1 = 0.5
pi2 = 0.5
mu_male_Height = 173.0
mu_male_Weight = 63.0
mu_female_Height = 162.0
mu_female_Weight = 52.0

# 假设男性和女性的身高和体重的标准差都是数据集中所有数据的标准差
sigma_male_Height = data['Height(cm)'].std()
sigma_male_Weight = data['Weight(kg)'].std()
sigma_female_Height = data['Height(cm)'].std()
sigma_female_Weight = data['Weight(kg)'].std()

# 迭代次数
max_iter = 1000


# EM算法
for i in range(max_iter):
    # E步：计算期望
    gamma_male = pi1 * norm.pdf(data['Height(cm)'], mu_male_Height, sigma_male_Height) * norm.pdf(data['Weight(kg)'], mu_male_Weight, sigma_male_Weight)
    gamma_female = pi2 * norm.pdf(data['Height(cm)'], mu_female_Height, sigma_female_Height) * norm.pdf(data['Weight(kg)'], mu_female_Weight, sigma_female_Weight)
    gamma_male /= (gamma_male + gamma_female )
    gamma_female = 1 - gamma_male


    # M步：最大化期望
    N_male = gamma_male.sum()
    N_female = gamma_female.sum()
    pi1 = N_male / (len(data) )
    pi2 = N_female / (len(data) )
    mu_male_Height = (gamma_male * data['Height(cm)']).sum() / N_male
    mu_female_Height = (gamma_female * data['Height(cm)']).sum() / N_female
    mu_male_Weight = (gamma_male * data['Weight(kg)']).sum() / N_male
    mu_female_Weight = (gamma_female * data['Weight(kg)']).sum() / N_female
    sigma_male_Height = np.sqrt((gamma_male * (data['Height(cm)'] - mu_male_Height)**2).sum() / N_male)
    sigma_female_Height = np.sqrt((gamma_female * (data['Height(cm)'] - mu_female_Height)**2).sum() / N_female)
    sigma_male_Weight = np.sqrt((gamma_male * (data['Weight(kg)'] - mu_male_Weight)**2).sum() / N_male)
    sigma_female_Weight = np.sqrt((gamma_female * (data['Weight(kg)'] - mu_female_Weight)**2).sum() / N_female)


print('男性身高均值:', mu_male_Height, '男性身高标准差:', sigma_male_Height)
print('女性身高均值:', mu_female_Height, '女性身高标准差:', sigma_female_Height)
print('男性体重均值:', mu_male_Weight, '男性体重标准差:', sigma_male_Weight)
print('女性体重均值:', mu_female_Weight, '女性体重标准差:', sigma_female_Weight)