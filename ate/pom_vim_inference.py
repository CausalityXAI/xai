#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
#%%
"""
Reference:
[1]: https://arxiv.org/pdf/2204.06030.pdf
"""
# np.random.seed(1)
# n = 1000
# p = 2
# X = np.random.uniform(low=-1, high=1, size=(n, p))
# logit = -0.4 * X[:, 0] + 0.1 * X[:, 0] * X[:, 1]
# prob = 1 / (1 + np.exp(-logit))
# treatment = np.random.binomial(n=1, p=prob)
# cate = (X[:, 0] ** 2) * (X[:, 0] + 7/5) + 25 * (X[:, 1] ** 2) / 9
# outcome = X[:, 0] * X[:, 1] + 2 * (X[:, 1] ** 2) - X[:, 0] + treatment * cate + np.random.normal(size=n)

# linear case
np.random.seed(1)
n = 1000
p = 5
X = np.random.uniform(low=-1, high=1, size=(n, p))
beta = np.array([[0.5, -0.4, 0.3, 0.2, -0.1]])
logit = X @ beta.T
prob = 1 / (1 + np.exp(-logit))
treatment = np.random.binomial(n=1, p=prob)
beta = np.array([[1, -2, -3, -4, 5]])
cate = X @ beta.T
beta = np.array([[-5, -4, 3, -2, 1]])
outcome = X @ beta.T + treatment * cate + np.random.normal(size=(n, 1))

data = np.concatenate([X, treatment, outcome], axis=1)
covariates = ['X{}'.format(i+1) for i in range(p)]
data = pd.DataFrame(data, columns=covariates + ['treatment', 'outcome'])
#%%
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

np.random.seed(0)
K = 2 # cross-fitted inference
m = data.shape[0] // K
index_list = [m * i for i in range(K)] + [data.shape[0]]

VIM = []
var = []
for subset in tqdm.tqdm(covariates):
    subset_complement = [x for x in covariates if x not in subset]
    
    psi = 0
    tau = 0
    for i in range(len(index_list) - 1):
        idx = index_list[i]
        train = data.iloc[index_list[i] : index_list[i+1]]
        test = pd.concat([data.iloc[: index_list[i]], data.iloc[index_list[i+1] : ]], axis=0)
        
        """Conditional mean: Q_0(a, x) = E_0[Y | A = a, X = x]"""
        # conditional_mean = linear_model.LinearRegression()
        # conditional_mean.fit(train[covariates + ['treatment']], train['outcome'])
        conditional_mean = RandomForestRegressor(random_state=0)
        conditional_mean.fit(train[covariates + ['treatment']], train['outcome'])
        
        """Propensity score: pi_0(a, x) = Pr_0[A = a | X = x]"""
        # propensity_score = linear_model.LogisticRegression(max_iter=10000)
        # propensity_score.fit(train[covariates], train['treatment'])
        propensity_score = RandomForestClassifier(random_state=0)
        propensity_score.fit(train[covariates], train['treatment'])
        
        """Treatment rule: f_0(x)"""
        test1 = pd.DataFrame.copy(test)
        test1['treatment'] = 1
        test0 = pd.DataFrame.copy(test)
        test0['treatment'] = 0
        treatment_rule = (conditional_mean.predict(test1[covariates + ['treatment']]) > 
                          conditional_mean.predict(test0[covariates + ['treatment']])).astype(int)
        
        """residual"""
        pred = conditional_mean.predict(train[covariates + ['treatment']])
        # conditional_mean_residual = linear_model.LinearRegression()
        # conditional_mean_residual.fit(train[subset_complement + ['treatment']], pred)
        conditional_mean_residual = RandomForestRegressor(random_state=0)
        conditional_mean_residual.fit(train[subset_complement + ['treatment']], pred)
        
        test1 = pd.DataFrame.copy(test)
        test1['treatment'] = 1
        test0 = pd.DataFrame.copy(test)
        test0['treatment'] = 0
        treatment_rule_residual = (conditional_mean_residual.predict(test1[subset_complement + ['treatment']]) > 
                                   conditional_mean_residual.predict(test0[subset_complement + ['treatment']])).astype(int)
        
        """EIF"""
        indicator = np.array((test['treatment'] == treatment_rule).astype(float))
        prob = (propensity_score.predict_proba(test[covariates]) * np.eye(2)[treatment_rule]).sum(axis=1)
        test_ = pd.DataFrame.copy(test)
        test_['treatment'] = treatment_rule
        EIF = (indicator / prob) * (test_['outcome'] - conditional_mean.predict(test_[covariates + ['treatment']]))
        EIF += conditional_mean.predict(test_[covariates + ['treatment']])
        predictiveness = conditional_mean.predict(test_[covariates + ['treatment']]).mean()
        EIF -= predictiveness
        
        """EIF residual"""
        indicator = np.array((test['treatment'] == treatment_rule_residual).astype(float))
        prob = (propensity_score.predict_proba(test[covariates]) * np.eye(2)[treatment_rule_residual]).sum(axis=1)
        test_ = pd.DataFrame.copy(test)
        test_['treatment'] = treatment_rule_residual
        EIF_residual = (indicator / prob) * (test_['outcome'] - conditional_mean.predict(test_[covariates + ['treatment']]))
        EIF_residual += conditional_mean.predict(test_[covariates + ['treatment']])
        predictiveness_residual = conditional_mean.predict(test_[covariates + ['treatment']]).mean()
        EIF_residual -= predictiveness_residual
        
        """VIM"""
        oracle = (predictiveness + EIF.mean()) - (predictiveness_residual + EIF_residual.mean())
        
        """asymptotic variance"""
        asymptotic_variance = ((EIF - EIF_residual) ** 2).mean()
        
        psi += oracle
        tau += asymptotic_variance
        
    VIM.append(psi / K)
    var.append(tau / K)
#%%
result = [(n,
            p - 1.96 * v / np.sqrt(data.shape[0]),
            p + 1.96 * v / np.sqrt(data.shape[0]),
            p) for p, v, n in zip(VIM, var, covariates)]
for name, lower, upper, v in result:
    print('Covariate: {}'.format(name))
    print('VIM: {}'.format(v))
    print('CI:', lower, upper)
    print()
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.errorbar([x[0] for x in result], 
             [x[3] for x in result], 
             [x[2] - x[3] for x in result], 
             linestyle='None', marker='o')
plt.ylabel('VIM')
plt.savefig('assets/VIM_toy.png')
#%%