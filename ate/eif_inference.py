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
from sklearn.linear_model import LogisticRegression
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
df = binary_t_df
df.columns
#%%
"""Propensity score"""
score = LogisticRegression()
score.fit(df[['age', 'proteinuria']], df['sodium'])
score.intercept_
score.coef_
#%%
"""conditional mean"""
cmean = LinearRegression()
cmean.fit(df[['sodium', 'age', 'proteinuria']], df['blood_pressure'])
cmean.intercept_
cmean.coef_
#%%
"""Efficient Influence Function"""
df['EIF'] = 0
estimated_psi = []
for T in [0, 1]:
    df_ = pd.DataFrame.copy(df)
    df_['sodium'] = T

    IP = 1 / score.predict_proba(df[['age', 'proteinuria']])[:, T] # inverse probability
    m = cmean.predict(df_[['sodium', 'age', 'proteinuria']]) # conditional mean with x
    IPW = IP * (df['blood_pressure'] - m)
    IPW = np.array(df['sodium'] == T).astype(float) * IPW + m
    psi = m.mean()
    phi = IPW - psi
    
    estimated_psi.append(psi)

    df['EIF'].loc[df['sodium'] == T] = phi[df['sodium'] == T]
#%%
"""inference and testing"""
psi = estimated_psi[1] - estimated_psi[0]
std = np.sqrt((df['EIF'] ** 2).mean())
CI = (psi - 1.96 * std / np.sqrt(n), psi + 1.96 * std / np.sqrt(n))
print('Confidence interval for ATE:', CI)

# assert CI[0] < beta1 and beta1 < CI[1]
#%%