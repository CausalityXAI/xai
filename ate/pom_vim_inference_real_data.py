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
[2]: https://www.rdocumentation.org/packages/speff2trial/versions/1.0.4/topics/ACTG175

We demonstrate our estimators on data from the AIDS Clinical Trials Group Protocol 175 (ACTG175) (Hammer et al., 1996), 
which evaluated 2139 patients infected with HIV whose CD4 T-cell count was in the range 200 to 500 mm^−3. 
(The data is available through the speff2trial package in R)

Patients were randomised to 4 treatment groups: 
(i) zidovudine (ZDV) monotherapy
(ii) ZDV+didanosine (ddI)
(iii) ZDV+zalcitabine
(iv) ddI monotherapy

Treatment groups (Lu et al. (2013); Cui et al. (2020))
A = 0: (iv), n = 561
A = 1: (ii), n = 522

outcome: CD4 count at 20±5 weeks as a continuous outcome, Y
12 covariates: 
5 continuous: age, weight, Karnofsky score, CD4 count, CD8 count
7 binary: sex, homosexual activity (y/n), race (white/non-white), 
symptomatic status (symptomatic/asymptomatic), history of intravenous drug use (y/n), 
hemophilia (y/n), and antiretroviral history (experienced/naive).
"""
data = pd.read_csv('/Users/anseunghwan/Documents/GitHub/xai/ate/assets/ACTG175.csv')
data.columns

# treatment groups
data = data.iloc[[x in [1, 3] for x in data['arms']]]
data['treatment'] = data['arms'].apply(lambda x: 0 if x == 3 else 1)

covariates = [
    'age', 'wtkg', 'karnof', 'cd40', 'cd80',
    'gender', 'homo', 'race', 'symptom', 'drugs', 'hemo', 'str2'
]
data = data[['cd420'] + ['treatment'] + covariates]
data.shape 
data = data.sample(frac=1).reset_index(drop=True) # shuffle
#%%
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# max_features = ['auto']
# min_samples_split = [2, 5]
# min_samples_leaf = [1, 2]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf}
# print(random_grid)

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
        # conditional_mean.fit(train[covariates + ['treatment']], train['cd420'])
        conditional_mean = RandomForestRegressor(random_state=0)
        conditional_mean.fit(train[covariates + ['treatment']], train['cd420'])
        # conditional_mean = GridSearchCV(estimator = RandomForestRegressor(), 
        #                                 param_grid = random_grid, 
        #                                 cv = 2, 
        #                                 verbose = 2)
        # conditional_mean.fit(train[covariates + ['treatment']], train['cd420'])
        
        """Propensity score: pi_0(a, x) = Pr_0[A = a | X = x]"""
        # propensity_score = linear_model.LogisticRegression(max_iter=10000)
        # propensity_score.fit(train[covariates], train['treatment'])
        propensity_score = RandomForestClassifier(random_state=0)
        propensity_score.fit(train[covariates], train['treatment'])
        # propensity_score = GridSearchCV(estimator = RandomForestClassifier(), 
        #                                 param_grid = random_grid, 
        #                                 cv = 2, 
        #                                 verbose = 2)
        # propensity_score.fit(train[covariates], train['treatment'])
        
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
        # conditional_mean_residual = GridSearchCV(estimator = RandomForestRegressor(), 
        #                                         param_grid = random_grid, 
        #                                         cv = 2, 
        #                                         verbose = 2)
        # conditional_mean_residual.fit(train[subset_complement + ['treatment']], pred)
        
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
        EIF = (indicator / prob) * (test_['cd420'] - conditional_mean.predict(test_[covariates + ['treatment']]))
        EIF += conditional_mean.predict(test_[covariates + ['treatment']])
        predictiveness = conditional_mean.predict(test_[covariates + ['treatment']]).mean()
        EIF -= predictiveness
        
        """EIF residual"""
        indicator = np.array((test['treatment'] == treatment_rule_residual).astype(float))
        prob = (propensity_score.predict_proba(test[covariates]) * np.eye(2)[treatment_rule_residual]).sum(axis=1)
        test_ = pd.DataFrame.copy(test)
        test_['treatment'] = treatment_rule_residual
        EIF_residual = (indicator / prob) * (test_['cd420'] - conditional_mean.predict(test_[covariates + ['treatment']]))
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
result_ = sorted([(x[0], x[3]) for x in result], key=lambda x:-x[1])

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot([x[0] for x in result_], 
        [x[1] for x in result_],
        linestyle='None', marker='o')
plt.ylabel('VIM')
plt.savefig('assets/VIM_ACTG175.png')
# plt.errorbar([x[0] for x in result], 
#              [x[3] for x in result], 
#              [x[2] - x[3] for x in result], 
#              linestyle='None', marker='o')
#%%
        # np.sqrt(metrics.mean_squared_error(test['cd420'], model.predict(test[covariates + ['treatment']])))

#%%