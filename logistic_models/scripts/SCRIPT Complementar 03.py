# Estimação de um modelo logístico binário pela função 'Logit.from_formula'
#('statsmodels')

modelo_atrasos2 = sm.Logit.from_formula('atrasado ~ dist + sem',
                                        data=df_atrasado).fit()

# Parâmetros do 'modelo_atrasos2'
modelo_atrasos2.summary()

# Loglike do 'modelo_atrasos2'
modelo_atrasos2.llf

#%%
# Cálculo do chi2 (análogo ao Teste F da estimação por OLS)

modelo_nulo = sm.Logit.from_formula('atrasado ~ 1',
                                        data=df_atrasado).fit()

# Parâmetros do 'modelo_nulo'
modelo_nulo.summary()

# Loglike do 'modelo_nulo'
modelo_nulo.llf

chi2 = -2*(modelo_nulo.llf - modelo_atrasos2.llf)
chi2

pvalue = stats.distributions.chi2.sf(chi2, 2)
pvalue

#%%
# Pseudo R² de McFadden (Prêmio Nobel de Economia em 2000)

pseudor2 = ((-2*modelo_nulo.llf)-(-2*modelo_atrasos2.llf))/(-2*modelo_nulo.llf)
pseudor2

#%%
# AIC (Akaike Info Criterion)
aic = -2*(modelo_atrasos2.llf) + 2*(3)
aic

modelo_atrasos2.aic

# BIC (Bayesian Info Criterion)
bic = -2*(modelo_atrasos2.llf) + 3*np.log(100)
bic

modelo_atrasos2.bic
