# MODELO APENAS COM ENDIVIDAMENTO

modelo_auxiliar1 = sm.OLS.from_formula('retorno ~ endividamento',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar1'
modelo_auxiliar1.summary()

#%%
# MODELO SEM O ENDIVIDAMENTO

modelo_auxiliar2 = sm.OLS.from_formula('retorno ~ disclosure + ativos +\
                                       liquidez', df_empresas).fit()

# Parâmetros do 'modelo_auxiliar2'
modelo_auxiliar2.summary()


#%%
# MODELO SEM O DISCLOSURE (MODELO FINAL LINEAR)

modelo_auxiliar3 = sm.OLS.from_formula('retorno ~ ativos +\
                                       liquidez', df_empresas).fit()

# Parâmetros do 'modelo_auxiliar3'
modelo_auxiliar3.summary()

#%%
# MODELO APENAS COM DISCLOSURE

modelo_auxiliar4 = sm.OLS.from_formula('retorno ~ disclosure',
                                       df_empresas).fit()

# Parâmetros do 'modelo_auxiliar4'
modelo_auxiliar4.summary()
