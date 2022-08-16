#%%
"""
Reference:
Estimating the causal effect of sodium on blood pressure in a simulated example
adapted from Luque-Fernandez et al. (2018):
    https://academic.oup.com/ije/article/48/2/640/5248195
"""
#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#%%
"""
generate dataset
"""
def generate_data(n=1000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3, binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
    proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
    hypertension = (blood_pressure >= 140).astype(int)  # not used, but could be used for binary outcomes
    return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,
                         'age': age, 'proteinuria': proteinuria})
#%%
n=10000
seed=0
beta1=1.05 # Simulation: so we know the true ATE is 1.05!
alpha1=0.4
alpha2=0.3
binary_treatment=True
binary_cutoff=3.5

binary_t_df = generate_data(beta1=1.05, alpha1=.4, alpha2=.3, binary_treatment=True, n=n)
#%%
def estimate_causal_effect(Xt, y, model=LinearRegression(), treatment_idx=0, regression_coef=False):
    """
    model assumption: linear model
    Y = alpha * T + beta * X
    
    Y: blood pressure
    T: sodium
    X: age, proteinuria
    
    if regression_coef = True:
        Y(1) - Y(0) = (alpha + beta * X) - (beta * X) = alpha
        (Severe limitations: the causal effect is the same for all individuals)
    else: # adjustment
        E[E[Y | T=1, X] - E[Y | T=0, X]]
    """
    model.fit(Xt, y)
    if regression_coef:
        return model.coef_[treatment_idx]
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1 # set treatment of population to 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0 # set treatment of population to 0
        return (model.predict(Xt1) - model.predict(Xt0)).mean()
#%%
df = binary_t_df

"""Linear regression coefficient estimates"""
ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
                                            df['blood_pressure'], treatment_idx=0,
                                            regression_coef=True)
print('# Regression Coefficient Estimates #')
print('ATE estimate adjusting for all covariates:', round(ate_est_adjust_all, 2))
print()

"""Adjustment formula estimates"""
ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
                                            df['blood_pressure'], treatment_idx=0)
print('# Adjustment Formula Estimates #')
print('ATE estimate adjusting for all covariates:', round(ate_est_adjust_all, 2))
print()
#%%
# """Linear regression coefficient estimates"""
# ate_est_naive = estimate_causal_effect(df[['sodium']], df['blood_pressure'], treatment_idx=0,
#                                         regression_coef=True)
# ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
#                                             df['blood_pressure'], treatment_idx=0,
#                                             regression_coef=True)
# ate_est_adjust_age = estimate_causal_effect(df[['sodium', 'age']], df['blood_pressure'],
#                                             regression_coef=True)
# print('# Regression Coefficient Estimates #')
# print('Naive ATE estimate:', ate_est_naive)
# print('ATE estimate adjusting for all covariates:', ate_est_adjust_all)
# print('ATE estimate adjusting for age:', ate_est_adjust_age)
# print()

# """Adjustment formula estimates"""
# ate_est_naive = estimate_causal_effect(df[['sodium']], df['blood_pressure'], treatment_idx=0)
# ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
#                                             df['blood_pressure'], treatment_idx=0)
# ate_est_adjust_age = estimate_causal_effect(df[['sodium', 'age']], df['blood_pressure'])
# print('# Adjustment Formula Estimates #')
# print('Naive ATE estimate:', ate_est_naive)
# print('ATE estimate adjusting for all covariates:', ate_est_adjust_all)
# print('ATE estimate adjusting for age:', ate_est_adjust_age)
# print()
#%%